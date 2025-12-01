"""Implement various linear algebra algorithms for low rank matrices."""

__all__ = ["svd_lowrank", "svd_lowrank_eig", "pca_lowrank", "svd_lowrank_eig_graph"]

from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import torch
from torch import _linalg_utils as _utils, Tensor
from torch.overrides import handle_torch_function, has_torch_function
from torch.autograd.grad_mode import no_grad as _no_grad

from torch import Tensor

from contextlib import contextmanager


@contextmanager
def _autocast_if_needed(enabled: bool, mp_dtype: Optional[torch.dtype]):
    if enabled and mp_dtype in (torch.bfloat16, torch.float16):
        with torch.cuda.amp.autocast(dtype=mp_dtype):
            yield
    else:
        yield


def _mm_proj(A: Tensor, B: Tensor, *, use_mp: bool, mp_dtype: Optional[torch.dtype]) -> Tensor:
    """
    仅用于投影的 GEMM：在 autocast 下做 mm，然后立即转回 A.dtype。
    始终返回与 A 相同 dtype 的张量，避免精度持续下降。
    """
    with _autocast_if_needed(use_mp, mp_dtype):
        C = torch.mm(A, B)  # 2D mm，性能更好
    if C.dtype != A.dtype:
        C = C.to(A.dtype)
    return C


def get_approximate_basis(
    A: Tensor,
    q: int,
    niter: Optional[int] = 2,
    M: Optional[Tensor] = None,
) -> Tensor:
    """Return tensor :math:`Q` with :math:`q` orthonormal columns such
    that :math:`Q Q^H A` approximates :math:`A`. If :math:`M` is
    specified, then :math:`Q` is such that :math:`Q Q^H (A - M)`
    approximates :math:`A - M`. without instantiating any tensors
    of the size of :math:`A` or :math:`M`.

    .. note:: The implementation is based on the Algorithm 4.4 from
              Halko et al., 2009.

    .. note:: For an adequate approximation of a k-rank matrix
              :math:`A`, where k is not known in advance but could be
              estimated, the number of :math:`Q` columns, q, can be
              choosen according to the following criteria: in general,
              :math:`k <= q <= min(2*k, m, n)`. For large low-rank
              matrices, take :math:`q = k + 5..10`.  If k is
              relatively small compared to :math:`min(m, n)`, choosing
              :math:`q = k + 0..2` may be sufficient.

    .. note:: To obtain repeatable results, reset the seed for the
              pseudorandom number generator

    Args::
        A (Tensor): the input tensor of size :math:`(*, m, n)`

        q (int): the dimension of subspace spanned by :math:`Q`
                 columns.

        niter (int, optional): the number of subspace iterations to
                               conduct; ``niter`` must be a
                               nonnegative integer. In most cases, the
                               default value 2 is more than enough.

        M (Tensor, optional): the input tensor's mean of size
                              :math:`(*, m, n)`.

    References::
        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <http://arxiv.org/abs/0909.4061>`_).
    """

    niter = 2 if niter is None else niter
    dtype = _utils.get_floating_dtype(A) if not A.is_complex() else A.dtype
    matmul = _utils.matmul

    R = torch.randn(A.shape[-1], q, dtype=dtype, device=A.device)

    # The following code could be made faster using torch.geqrf + torch.ormqr
    # but geqrf is not differentiable

    X = matmul(A, R)
    if M is not None:
        X = X - matmul(M, R)
    Q = torch.linalg.qr(X).Q
    for _ in range(niter):
        X = matmul(A.mH, Q)
        if M is not None:
            X = X - matmul(M.mH, Q)
        Q = torch.linalg.qr(X).Q
        X = matmul(A, Q)
        if M is not None:
            X = X - matmul(M, Q)
        Q = torch.linalg.qr(X).Q
    return Q


@_no_grad()
def _cholqr(Y: torch.Tensor) -> torch.Tensor:
    G = Y.mH @ Y
    try:
        L, info = torch.linalg.cholesky_ex(G, upper=False)
        if int(info.item()) != 0:
            return torch.linalg.qr(Y, mode="reduced").Q
        T = torch.linalg.solve_triangular(L, Y.mH, upper=False)
        return T.mH
    except RuntimeError:
        return torch.linalg.qr(Y, mode="reduced").Q


def get_approximate_basis_new(
    A: Tensor,
    q: int,
    niter: Optional[int] = 2,
    M: Optional[Tensor] = None,
) -> Tensor:
    """
    优化版 range finder：
      - 正交化用 cholQR（比 QR 快，q≪m 场景收益明显）
      - 每轮 power 迭代只做一次正交化：初始化 1 次 + 每轮 1 次
      - 与原版广播/复数/稀疏处理保持一致（这里针对 dense）
    建议：速度优先传 niter=0 并用 q=k+8/16 做 oversampling。
    """
    niter = 0 if niter is None else int(niter)
    dtype = _utils.get_floating_dtype(A) if not A.is_complex() else A.dtype
    matmul = _utils.matmul

    # 采样矩阵 Ω：形状 (n, q)
    R = torch.randn(A.shape[-1], q, dtype=dtype, device=A.device)

    # 初次采样：Y = (A - M) @ Ω
    Y = matmul(A, R)
    if M is not None:
        Y = Y - matmul(M, R)

    # 初次正交：Q0 = cholQR(Y)
    Q = _cholqr(Y)

    # 合并 power-iteration：每轮只正交一次
    # 经典：Q ← orth( A (A^H Q) )，若有 M：A→(A-M)
    for _ in range(niter):
        Z = matmul(A.mH, Q)  # Z = A^H Q
        if M is not None:
            Z = Z - matmul(M.mH, Q)  # Z -= M^H Q
        Y = matmul(A, Z)  # Y = A Z
        if M is not None:
            Y = Y - matmul(M, Z)  # Y -= M Z
        Q = _cholqr(Y)  # 仅在每轮末尾做一次正交

    return Q


def svd_lowrank(
    A: Tensor,
    q: Optional[int] = 6,
    niter: Optional[int] = 2,
    M: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Return the singular value decomposition ``(U, S, V)`` of a matrix,
    batches of matrices, or a sparse matrix :math:`A` such that
    :math:`A \approx U \operatorname{diag}(S) V^{\text{H}}`. In case :math:`M` is given, then
    SVD is computed for the matrix :math:`A - M`.

    .. note:: The implementation is based on the Algorithm 5.1 from
              Halko et al., 2009.

    .. note:: For an adequate approximation of a k-rank matrix
              :math:`A`, where k is not known in advance but could be
              estimated, the number of :math:`Q` columns, q, can be
              choosen according to the following criteria: in general,
              :math:`k <= q <= min(2*k, m, n)`. For large low-rank
              matrices, take :math:`q = k + 5..10`.  If k is
              relatively small compared to :math:`min(m, n)`, choosing
              :math:`q = k + 0..2` may be sufficient.

    .. note:: This is a randomized method. To obtain repeatable results,
              set the seed for the pseudorandom number generator

    .. note:: In general, use the full-rank SVD implementation
              :func:`torch.linalg.svd` for dense matrices due to its 10x
              higher performance characteristics. The low-rank SVD
              will be useful for huge sparse matrices that
              :func:`torch.linalg.svd` cannot handle.

    Args::
        A (Tensor): the input tensor of size :math:`(*, m, n)`

        q (int, optional): a slightly overestimated rank of A.

        niter (int, optional): the number of subspace iterations to
                               conduct; niter must be a nonnegative
                               integer, and defaults to 2

        M (Tensor, optional): the input tensor's mean of size
                              :math:`(*, m, n)`, which will be broadcasted
                              to the size of A in this function.

    References::
        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <https://arxiv.org/abs/0909.4061>`_).

    """
    if not torch.jit.is_scripting():
        tensor_ops = (A, M)
        if not set(map(type, tensor_ops)).issubset(
            (torch.Tensor, type(None))
        ) and has_torch_function(tensor_ops):
            return handle_torch_function(svd_lowrank, tensor_ops, A, q=q, niter=niter, M=M)
    return _svd_lowrank(A, q=q, niter=niter, M=M)


def _svd_lowrank(
    A: Tensor,
    q: Optional[int] = 6,
    niter: Optional[int] = 2,
    M: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    # Algorithm 5.1 in Halko et al., 2009

    q = 6 if q is None else q
    m, n = A.shape[-2:]
    matmul = _utils.matmul
    if M is not None:
        M = M.broadcast_to(A.size())

    # Assume that A is tall
    if m < n:
        A = A.mH
        if M is not None:
            M = M.mH

    Q = get_approximate_basis(A, q, niter=niter, M=M)
    B = matmul(Q.mH, A)
    if M is not None:
        B = B - matmul(Q.mH, M)
    U, S, Vh = torch.linalg.svd(B, full_matrices=False)
    V = Vh.mH
    U = Q.matmul(U)

    if m < n:
        U, V = V, U

    return U, S, V


# 复用你现有的 get_approximate_basis(A, q, niter, M)


def svd_lowrank_eig(
    A: Tensor,
    q: Optional[int] = 6,
    niter: Optional[int] = 2,
    M: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    低秩随机 SVD（RandSVD）与 torch.svd_lowrank 同接口，
    但将小矩阵 SVD 换为 EIGH→eig-SVD。
    返回 (U, S, V)，满足 A ≈ U diag(S) V^H；若传入 M，则分解 A - M。
    """
    if not torch.jit.is_scripting():
        tensor_ops = (A, M)
        if not set(map(type, tensor_ops)).issubset(
            (torch.Tensor, type(None))
        ) and has_torch_function(tensor_ops):
            return handle_torch_function(svd_lowrank_eig, tensor_ops, A, q=q, niter=niter, M=M)
    return _svd_lowrank_eig(A, q=q, niter=niter, M=M)


def _svd_lowrank_eig(
    A: Tensor,
    q: Optional[int] = 6,
    niter: Optional[int] = 2,
    M: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    eig-SVD 小头（B 路线）优化版：
      Q = range(A) → B = Q^H A → G = B B^H → eigh(G)
      S = sqrt(λ)（降序），U = Q Ue，V = B^H (Ue · S^{-1})
    """
    q = 6 if q is None else q
    m, n = A.shape[-2:]
    if M is not None:
        M = M.broadcast_to(A.size())

    flipped = False
    if m < n:
        A = A.mH
        if M is not None:
            M = M.mH
        m, n = n, m
        flipped = True

    # 1) 子空间
    Q = get_approximate_basis_new(A, q, niter=niter, M=M)  # (m, q)

    # 2) B = Q^H A  （2D mm + 仅在必要时 contiguous）
    QH = Q.mH if Q.is_contiguous() else Q.mH.contiguous()  # (q, m)
    A_c = A if A.is_contiguous() else A.contiguous()  # (m, n)
    B = torch.mm(QH, A_c)  # (q, n)
    if M is not None:
        Mc = M if M.is_contiguous() else M.contiguous()
        B.sub_(torch.mm(QH, Mc))  # B -= Q^H M

    # 3) G = B B^H；eigh（升序）→ S = sqrt(λ)，降序 & 同步列重排
    G = torch.mm(B, B.mH)  # (q, q)
    evals, Ue = torch.linalg.eigh(G)  # evals 升序, Ue:(q,q)

    # 就地 clamp+sqrt，复用 evals 存 S
    evals.clamp_min_(0.0).sqrt_()
    S = evals  # (q,)

    # 降序重排（q 小，这一步分配/拷贝很便宜）
    S = S.flip(-1)  # (q,)
    Ue = Ue.flip(-1)  # (q,q) 仅翻列，与 S 对齐

    # 4) U = Q Ue
    U = torch.mm(Q, Ue)  # (m, q)

    # 5) Ue 预缩放列（阈值保护）→ V = B^H @ Ue_scaled
    finfo = torch.finfo(A.real.dtype) if A.is_complex() else torch.finfo(A.dtype)
    tol = finfo.eps * q * S.amax()  # 标量
    Sinv = torch.where(S > tol, S.reciprocal(), torch.zeros_like(S))
    Ue.mul_(Sinv.unsqueeze(0))  # 就地列缩放
    V = torch.mm(B.mH, Ue)  # (n, q)

    if flipped:
        U, V = V, U
    return U, S, V


# ======== BEGIN: Two-Graph niter=1 with graph-safe cholQR (no info.item, no QR fallback) ========

# ===== 新的 Two-Graph 图算子：与 test 脚本逻辑一致 =====
from typing import Optional, Tuple, Dict


class _TwoGraphEigSVD_Niter1:
    """
    逻辑与你 test 脚本一致：
      front 图捕获：
        - RF-Init:      Y = A R → cholQR → Q
        - RF-PowerStep: Z = A^H Q → Y = A Z → cholQR → Q
        - Build:        B = Q^H A,  G = B B^H
      eigh(G) 在图外（q×q 非常小）
      back 图捕获：
        - U = Q Ue；Ue *= Sinv[None,:]；V = B^H Ue

    仅支持 niter=1（与 test 脚本相同）。
    """

    def __init__(self, m: int, n: int, q: int, device: torch.device, dtype: torch.dtype):
        self.m, self.n, self.q = int(m), int(n), int(q)
        self.device, self.dtype = device, dtype

        # 持久缓冲：固定地址/stride 以便在图内复用
        self.A = torch.empty(m, n, device=device, dtype=dtype)  # in
        self.R = torch.empty(n, q, device=device, dtype=dtype)  # in（每轮随机采样，拷进来）
        self.Y = torch.empty(m, q, device=device, dtype=dtype)  # tmp
        self.Z = torch.empty(n, q, device=device, dtype=dtype)  # tmp
        self.Q = torch.empty(m, q, device=device, dtype=dtype)  # out(front)
        self.B = torch.empty(q, n, device=device, dtype=dtype)  # out(front)
        self.G = torch.empty(q, q, device=device, dtype=dtype)  # out(front)

        self.Ue = torch.empty(q, q, device=device, dtype=dtype)  # in(back)
        self.S = torch.empty(q, device=device, dtype=dtype)  # in(back)
        self.Sinv = torch.empty(q, device=device, dtype=dtype)  # in(back)

        self.U = torch.empty(m, q, device=device, dtype=dtype)  # out(back)
        self.V = torch.empty(n, q, device=device, dtype=dtype)  # out(back)

        # 两段图 + 专用捕获流（避免 cuSOLVER 在 capture 中首用）
        self._g_front = torch.cuda.CUDAGraph()
        self._g_back = torch.cuda.CUDAGraph()
        self._cap_stream = torch.cuda.Stream()
        self._captured = False

    # —— 图内 cholQR：与 test 脚本一致，不做 info.item 检查、也不回退 QR ——
    def _cholqr_inline_into_Q(self):
        # G = Y^H Y
        torch.mm(self.Y.mH, self.Y, out=self.G)
        # cholesky_ex（返回(L,info)，但我们仅取 L；不做 .item() 访问）
        L, _info = torch.linalg.cholesky_ex(self.G, upper=False)
        # solve_triangular：L T = Y^H
        T = torch.linalg.solve_triangular(L, self.Y.mH, upper=False)
        # Q = T^H
        self.Q.copy_(T.mH)

    def capture(self):
        if self._captured:
            return

        pool = torch.cuda.graphs.graph_pool_handle()

        # —— 关键：在“同一条私有流”上做完整预热 + 显式 cuSOLVER 预热 ——
        with torch.cuda.stream(self._cap_stream):
            # 预热 GEMM 路径
            self.A.zero_()
            self.R.zero_()

            # RF-Init 预热
            torch.mm(self.A, self.R, out=self.Y)
            self._cholqr_inline_into_Q()

            # RF-PowerStep 预热（niter=1）
            torch.mm(self.A.mH, self.Q, out=self.Z)
            torch.mm(self.A, self.Z, out=self.Y)
            self._cholqr_inline_into_Q()

            # Build 预热
            torch.mm(self.Q.mH, self.A, out=self.B)
            torch.mm(self.B, self.B.mH, out=self.G)

            # Back 预热
            self.Ue.zero_()
            self.S.fill_(1.0)
            self.Sinv.fill_(1.0)
            torch.mm(self.Q, self.Ue, out=self.U)
            self.Ue.mul_(self.Sinv.unsqueeze(0))
            torch.mm(self.B.mH, self.Ue, out=self.V)

            # 显式 cuSOLVER 预热：保证 cholesky_ex 已在本流完成初始化
            self.G.copy_(torch.eye(self.q, device=self.device, dtype=self.dtype))
            _ = torch.linalg.cholesky_ex(self.G, upper=False)

        # 等待私有流完成所有预热（含 cuSOLVER 懒初始化）
        self._cap_stream.synchronize()

        # —— 用同一条流做 capture：front & back ——
        with torch.cuda.stream(self._cap_stream):
            # front: RF-Init + RF-PowerStep(niter=1) + Build
            with torch.cuda.graph(self._g_front, pool=pool):
                # Y = A R
                torch.mm(self.A, self.R, out=self.Y)
                self._cholqr_inline_into_Q()  # Q0

                # Z = A^H Q; Y = A Z; cholQR -> Q
                torch.mm(self.A.mH, self.Q, out=self.Z)
                torch.mm(self.A, self.Z, out=self.Y)
                self._cholqr_inline_into_Q()  # Q1

                # Build
                torch.mm(self.Q.mH, self.A, out=self.B)  # B = Q^H A
                torch.mm(self.B, self.B.mH, out=self.G)  # G = B B^H

            # back: U、缩放、V
            with torch.cuda.graph(self._g_back, pool=pool):
                torch.mm(self.Q, self.Ue, out=self.U)  # U = Q Ue
                self.Ue.mul_(self.Sinv.unsqueeze(0))  # Ue *= Sinv[None,:]
                torch.mm(self.B.mH, self.Ue, out=self.V)  # V = B^H Ue

        self._captured = True

    @_no_grad()
    def run_once(self, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 拷入 A；R 每次随机采样
        self.A.copy_(A)
        torch.randn(self.R.shape, out=self.R)

        # front：完成 RF(Init+1) + Build，得到 Q、B、G
        self._g_front.replay()

        # eigh(G) 在图外（q×q 小矩阵）
        # —— 原来是：evals, Ue_local = torch.linalg.eigh(self.G)
        Gh = (self.G + (self.G.mH if self.G.is_complex() else self.G.T)) * 0.5
        tau = torch.finfo(Gh.dtype).eps * torch.diagonal(Gh, 0, -2, -1).abs().mean()
        evals, Ue_local = torch.linalg.eigh(
            Gh + tau * torch.eye(self.q, dtype=Gh.dtype, device=Gh.device)
        )
        evals.clamp_min_(0.0).sqrt_()
        S = evals.flip(-1)
        Ue_local = Ue_local.flip(-1)

        # tol & Sinv（与 test 脚本一致）
        finfo = torch.finfo(self.dtype if not A.is_complex() else A.real.dtype)
        tol = finfo.eps * self.q * S.amax()
        Sinv = torch.where(S > tol, S.reciprocal(), torch.zeros_like(S))

        # 写入 back 图缓冲
        self.S.copy_(S)
        self.Sinv.copy_(Sinv)
        self.Ue.copy_(Ue_local)

        # back：U、V
        self._g_back.replay()

        return self.U.clone(), self.S.clone(), self.V.clone()


# —— Engine 缓存 ——（device, dtype, m, n, q）→ 引擎
_ENGINE_CACHE_TWO_GRAPH: Dict[tuple, _TwoGraphEigSVD_Niter1] = {}


def _get_two_graph_engine(
    m: int, n: int, q: int, dev: torch.device, dt: torch.dtype
) -> _TwoGraphEigSVD_Niter1:
    key = (dev.index if dev.type == "cuda" else -1, str(dt), int(m), int(n), int(q))
    eng = _ENGINE_CACHE_TWO_GRAPH.get(key)
    if eng is None:
        eng = _TwoGraphEigSVD_Niter1(m, n, q, dev, dt)
        eng.capture()
        _ENGINE_CACHE_TWO_GRAPH[key] = eng
    return eng


@_no_grad()
def svd_lowrank_eig_graph(
    A: torch.Tensor,
    q: Optional[int] = 6,
    niter: Optional[int] = 1,
    M: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    与 test 脚本一致的图版：
      - 仅支持 niter=1；M 必须为 None（保持路径一致）
      - 若 m < n，则在 A^H 上运行，再交换 U/V 返回（与常规实现一致）
      - front/back 两段图； cholQR 在图内；eigh 在图外
    """
    if A.device.type != "cuda":
        raise RuntimeError("svd_lowrank_eig_graph 仅在 CUDA 上启用（与基准一致）")
    if M is not None:
        raise NotImplementedError("该图版与基准一致：不支持 M（中心化项）")
    q = 6 if q is None else int(q)
    if niter is None:
        niter = 1
    if int(niter) != 1:
        raise NotImplementedError("该图版与基准一致：仅支持 niter=1")

    m, n = A.shape[-2:]
    flipped = False
    A_use = A
    if m < n:
        # 与常规实现一致：在较大的维度上做（A^H），最后交换 U/V
        A_use = A.mH
        m, n = n, m
        flipped = True

    eng = _get_two_graph_engine(m, n, q, A_use.device, A_use.dtype)
    U, S, V = eng.run_once(A_use)

    if flipped:
        U, V = V, U
    return U, S, V


# ======== END: Two-Graph niter=1 with graph-safe cholQR ========


# ======== BEGIN: Two-Graph niter=1 Pipelined with async CPU eig ========
class _TwoGraphEigSVD_Niter1_Pipelined:
    def __init__(self, m: int, n: int, q: int, device: torch.device, dtype: torch.dtype):
        self.m, self.n, self.q = int(m), int(n), int(q)
        self.device, self.dtype = device, dtype

        # device buffers (persisted) - 与原实现一致
        self.A = torch.empty(m, n, device=device, dtype=dtype)  # in
        self.R = torch.empty(n, q, device=device, dtype=dtype)  # in（每轮随机采样，拷进来）
        self.Y = torch.empty(m, q, device=device, dtype=dtype)  # tmp
        self.Z = torch.empty(n, q, device=device, dtype=dtype)  # tmp
        self.Q = torch.empty(m, q, device=device, dtype=dtype)  # out(front)
        self.B = torch.empty(q, n, device=device, dtype=dtype)  # out(front)
        self.G = torch.empty(q, q, device=device, dtype=dtype)  # out(front)
        self.G_stable = torch.finfo(self.G.dtype).eps * torch.eye(
            self.G.size(-1), device=self.G.device, dtype=self.G.dtype
        )

        self.Ue = torch.empty(q, q, device=device, dtype=dtype)  # in(back)
        self.S = torch.empty(q, device=device, dtype=dtype)  # in(back)
        self.Sinv = torch.empty(q, device=device, dtype=dtype)  # in(back)

        self.U = torch.empty(m, q, device=device, dtype=dtype)  # out(back)
        self.V = torch.empty(n, q, device=device, dtype=dtype)  # out(back)

        # CUDA Graphs + stream + capture flag (跟你原始实现一致)
        self._g_front = torch.cuda.CUDAGraph()
        self._g_back = torch.cuda.CUDAGraph()
        self._cap_stream = torch.cuda.Stream()
        self._captured = False

        # --- 新增：线程池 + pinned cpu buffer 用于异步 cpu eig ---
        self._executor = ThreadPoolExecutor(max_workers=1)
        # pinned CPU buffer 用于 device->host 非阻塞拷贝（异步 DMA）
        self.G_cpu = torch.empty((q, q), device="cpu", pin_memory=True)
        self.Gs = torch.eye(q, device="cpu")
        # 存放 CPU 计算结果的临时变量（on CPU）
        self._cpu_future = None  # concurrent.futures.Future

    # 原有的 cholQR inline（保持不变）
    def _cholqr_inline_into_Q(self):
        # torch.mm(self.Y.mH, self.Y, out=self.G)
        # L, _info = torch.linalg.cholesky_ex(self.G + self.G_stable, upper=False)
        # T = torch.linalg.solve_triangular(L, self.Y.mH, upper=False)
        # self.Q.copy_(T.mH)

        # torch.mm(self.Y.mH, self.Y, out=self.G)
        # L, _info = torch.linalg.cholesky_ex(self.G + self.G_stable, upper=True)
        # T = torch.linalg.solve_triangular(L, self.Y, upper=True, left=False)
        # self.Q.copy_(T)

        torch.mm(self.Y.mH, self.Y, out=self.G)
        self.G.add_(self.G_stable)
        L, _info = torch.linalg.cholesky_ex(self.G, upper=True)
        torch.linalg.solve_triangular(L, self.Y, upper=True, left=False, out=self.Q)

        # torch.addmm(self.G_stable, self.Y.T, self.Y, out=self.G)
        # torch.linalg.cholesky_ex(self.G, upper=True, out=(self.Ue, self._info))
        # torch.linalg.solve_triangular(self.Ue, self.Y, upper=True, left=False, out=self.Q)

    def capture(self):
        if self._captured:
            return

        pool = torch.cuda.graphs.graph_pool_handle()

        # 预热：同原实现，在专用流上做 cuSOLVER / kernel 预热
        with torch.cuda.stream(self._cap_stream):
            self.A.zero_()
            self.R.zero_()
            # RF-Init 预热
            torch.mm(self.A, self.R, out=self.Y)
            self._cholqr_inline_into_Q()
            # RF-PowerStep 预热
            torch.mm(self.A.mH, self.Q, out=self.Z)
            torch.mm(self.A, self.Z, out=self.Y)
            self._cholqr_inline_into_Q()
            # Build 预热
            torch.mm(self.Q.mH, self.A, out=self.B)
            torch.mm(self.B, self.B.mH, out=self.G)
            # Back 预热
            self.Ue.zero_()
            self.S.fill_(1.0)
            self.Sinv.fill_(1.0)
            torch.mm(self.Q, self.Ue, out=self.U)
            self.Ue.mul_(self.Sinv.unsqueeze(0))
            torch.mm(self.B.mH, self.Ue, out=self.V)
            # cuSOLVER explicit warmup
            self.G.copy_(torch.eye(self.q, device=self.device, dtype=self.dtype))
            _ = torch.linalg.cholesky_ex(self.G, upper=False)
        self._cap_stream.synchronize()

        # Capture the graphs (front + back)
        with torch.cuda.stream(self._cap_stream):
            with torch.cuda.graph(self._g_front, pool=pool):
                # front: init + powerstep + build
                torch.mm(self.A, self.R, out=self.Y)
                self._cholqr_inline_into_Q()  # Q0
                torch.mm(self.A.mH, self.Q, out=self.Z)
                torch.mm(self.A, self.Z, out=self.Y)
                self._cholqr_inline_into_Q()  # Q1
                torch.mm(self.Q.mH, self.A, out=self.B)
                torch.mm(self.B, self.B.mH, out=self.G)
                self.G.add_(self.G_stable)

            with torch.cuda.graph(self._g_back, pool=pool):
                # back: U, scale, V
                torch.mm(self.Q, self.Ue, out=self.U)
                self.Ue.mul_(self.Sinv.unsqueeze(0))
                torch.mm(self.B.mH, self.Ue, out=self.V)

        self._captured = True

    # CPU-side helper: do eigh on pinned CPU buffer
    def _cpu_eigh_and_process(self, G_cpu: torch.Tensor):
        # G_cpu is pinned CPU tensor; the device->host copy may still be in flight:
        # reading it will implicitly wait until DMA completes.

        # evals, Ue = torch.linalg.eigh(G_cpu)   # CPU LAPACK

        # Gh = (G_cpu + (G_cpu.mH if self.G.is_complex() else G_cpu.T)) * 0.5
        # tau = torch.finfo(Gh.dtype).eps * torch.diagonal(Gh, 0, -2, -1).abs().mean()
        try:
            # evals, Ue = torch.linalg.eigh(Gh + tau * torch.eye(self.q, dtype=Gh.dtype, device=Gh.device))
            # torch.eye(self.q, out=self.Gs)
            evals, Ue = torch.linalg.eigh(G_cpu)
        except:
            print(f"A is NaN: {self.A.isnan().any()}")
            print(f"B is NaN: {self.B.isnan().any()}")
            print(f"Q is NaN: {self.Q.isnan().any()}")
            print(f"Z is NaN: {self.Z.isnan().any()}")
            print(f"R is NaN: {self.R.isnan().any()}")
            raise
        evals.clamp_min_(0.0).sqrt_()
        S = evals.flip(-1)
        Ue = Ue.flip(-1)

        finfo = torch.finfo(G_cpu.dtype)
        tol = finfo.eps * self.q * S.amax()
        Sinv = torch.where(S > tol, S.reciprocal(), torch.zeros_like(S))
        return S, Sinv, Ue  # all CPU tensors (S: (q,), Sinv:(q,), Ue:(q,q))

    def start_front_and_submit(self, A: torch.Tensor):
        """
        拷入输入 A，执行 front graph（replay），把 G 异步拷到 pinned CPU，
        然后在 DMA 完成后异步启动 CPU eigh。
        """
        assert A.shape == (self.m, self.n)
        self.A.copy_(A)
        torch.randn(self.R.shape, out=self.R)

        # 在 front stream 上执行前段 + 异步拷贝
        with torch.cuda.stream(self._cap_stream):
            self._g_front.replay()
            self.G_cpu.copy_(self.G, non_blocking=True)
            # 记录一个事件，表示拷贝完成
            copy_done = torch.cuda.Event()
            copy_done.record(self._cap_stream)

        # 定义 CPU 线程执行函数：先等 GPU 事件，再计算 eig
        def _wait_and_eigh(event: torch.cuda.Event, G_cpu: torch.Tensor):
            # 等待 GPU 拷贝真正完成
            event.synchronize()
            return self._cpu_eigh_and_process(G_cpu)

        # 在线程池中异步提交
        self._cpu_future = self._executor.submit(_wait_and_eigh, copy_done, self.G_cpu)

    def finish_back_and_replay(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        等待之前提交的 CPU eig 完成，把结果拷回 GPU（小量异步 copy），
        replay back graph（GPU），并返回 U,S,V（device tensors）。
        """
        # 确保有 pending CPU task
        if self._cpu_future is None:
            raise RuntimeError("No pending CPU eig task. Call start_front_and_submit() first.")

        # 等待 CPU 计算完成（线程池）
        S_cpu, Sinv_cpu, Ue_cpu = self._cpu_future.result()
        self._cpu_future = None

        with torch.cuda.stream(self._cap_stream):
            self.S.copy_(S_cpu, non_blocking=True)
            self.Sinv.copy_(Sinv_cpu, non_blocking=True)
            self.Ue.copy_(Ue_cpu, non_blocking=True)
            self._g_back.replay()

        # 确保 GPU back 结果可用
        torch.cuda.synchronize(self._cap_stream)

        # 返回 device tensors（可直接使用）
        return self.U.clone(), self.S.clone(), self.V.clone()

    # ---- convenience synchronous run (原 run_once 语义) ----
    def run_once(self, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        同步接口：按原 run_once 语义执行（front -> cpu eig -> back），
        但内部利用 pinned copy + threadpool (仍然阻塞直到完成)。
        """
        self.start_front_and_submit(A)
        return self.finish_back_and_replay()


# —— Pipelined Engine 缓存 ——（device, dtype, m, n, q）→ 引擎
_ENGINE_CACHE_TWO_GRAPH_PIPELINED: Dict[tuple, _TwoGraphEigSVD_Niter1_Pipelined] = {}


def _get_two_graph_pipelined_engine(
    m: int, n: int, q: int, dev: torch.device, dt: torch.dtype
) -> _TwoGraphEigSVD_Niter1_Pipelined:
    key = (dev.index if dev.type == "cuda" else -1, str(dt), int(m), int(n), int(q))
    eng = _ENGINE_CACHE_TWO_GRAPH_PIPELINED.get(key)
    if eng is None:
        eng = _TwoGraphEigSVD_Niter1_Pipelined(m, n, q, dev, dt)
        eng.capture()
        _ENGINE_CACHE_TWO_GRAPH_PIPELINED[key] = eng
    return eng


@_no_grad()
def svd_lowrank_eig_graph_pipelined(
    A: torch.Tensor,
    q: Optional[int] = 6,
    niter: Optional[int] = 1,
    M: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pipelined 图版（与 test 脚本一致）：
      - 仅支持 niter=1；M 必须为 None（保持路径一致）
      - 若 m < n，则在 A^H 上运行，再交换 U/V 返回（与常规实现一致）
      - front/back 两段图； cholQR 在图内；eigh 在图外，异步 CPU 计算
    """
    if A.device.type != "cuda":
        raise RuntimeError("svd_lowrank_eig_graph_pipelined 仅在 CUDA 上启用（与基准一致）")
    if M is not None:
        raise NotImplementedError("该图版与基准一致：不支持 M（中心化项）")
    q = 6 if q is None else int(q)
    if niter is None:
        niter = 1
    if int(niter) != 1:
        raise NotImplementedError("该图版与基准一致：仅支持 niter=1")

    m, n = A.shape[-2:]
    flipped = False
    A_use = A
    if m < n:
        A_use = A.mH
        m, n = n, m
        flipped = True

    eng = _get_two_graph_pipelined_engine(m, n, q, A_use.device, A_use.dtype)
    U, S, V = eng.run_once(A_use)

    if flipped:
        U, V = V, U
    return U, S, V


# ======== END: Two-Graph niter=1 Pipelined with async CPU eig ========


def pca_lowrank(
    A: Tensor,
    q: Optional[int] = None,
    center: bool = True,
    niter: int = 2,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Performs linear Principal Component Analysis (PCA) on a low-rank
    matrix, batches of such matrices, or sparse matrix.

    This function returns a namedtuple ``(U, S, V)`` which is the
    nearly optimal approximation of a singular value decomposition of
    a centered matrix :math:`A` such that :math:`A \approx U \operatorname{diag}(S) V^{\text{H}}`

    .. note:: The relation of ``(U, S, V)`` to PCA is as follows:

                - :math:`A` is a data matrix with ``m`` samples and
                  ``n`` features

                - the :math:`V` columns represent the principal directions

                - :math:`S ** 2 / (m - 1)` contains the eigenvalues of
                  :math:`A^T A / (m - 1)` which is the covariance of
                  ``A`` when ``center=True`` is provided.

                - ``matmul(A, V[:, :k])`` projects data to the first k
                  principal components

    .. note:: Different from the standard SVD, the size of returned
              matrices depend on the specified rank and q
              values as follows:

                - :math:`U` is m x q matrix

                - :math:`S` is q-vector

                - :math:`V` is n x q matrix

    .. note:: To obtain repeatable results, reset the seed for the
              pseudorandom number generator

    Args:

        A (Tensor): the input tensor of size :math:`(*, m, n)`

        q (int, optional): a slightly overestimated rank of
                           :math:`A`. By default, ``q = min(6, m,
                           n)``.

        center (bool, optional): if True, center the input tensor,
                                 otherwise, assume that the input is
                                 centered.

        niter (int, optional): the number of subspace iterations to
                               conduct; niter must be a nonnegative
                               integer, and defaults to 2.

    References::

        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <http://arxiv.org/abs/0909.4061>`_).

    """

    if not torch.jit.is_scripting():
        if type(A) is not torch.Tensor and has_torch_function((A,)):
            return handle_torch_function(pca_lowrank, (A,), A, q=q, center=center, niter=niter)

    (m, n) = A.shape[-2:]

    if q is None:
        q = min(6, m, n)
    elif not (q >= 0 and q <= min(m, n)):
        raise ValueError(
            f"q(={q}) must be non-negative integer and not greater than min(m, n)={min(m, n)}"
        )
    if not (niter >= 0):
        raise ValueError(f"niter(={niter}) must be non-negative integer")

    dtype = _utils.get_floating_dtype(A)

    if not center:
        return _svd_lowrank(A, q, niter=niter, M=None)

    if _utils.is_sparse(A):
        if len(A.shape) != 2:
            raise ValueError("pca_lowrank input is expected to be 2-dimensional tensor")
        c = torch.sparse.sum(A, dim=(-2,)) / m
        # reshape c
        column_indices = c.indices()[0]
        indices = torch.zeros(
            2,
            len(column_indices),
            dtype=column_indices.dtype,
            device=column_indices.device,
        )
        indices[0] = column_indices
        C_t = torch.sparse_coo_tensor(indices, c.values(), (n, 1), dtype=dtype, device=A.device)

        ones_m1_t = torch.ones(A.shape[:-2] + (1, m), dtype=dtype, device=A.device)
        M = torch.sparse.mm(C_t, ones_m1_t).mT
        return _svd_lowrank(A, q, niter=niter, M=M)
    else:
        C = A.mean(dim=(-2,), keepdim=True)
        return _svd_lowrank(A - C, q, niter=niter, M=None)
