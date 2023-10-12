# Need to be in seperate file as it cannot have
# from __future__ import annotations

from typing import TYPE_CHECKING, Any

if __debug__ or TYPE_CHECKING:

    def exec_saving_source(
        source: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
    ):
        """Equivalent to exec, but allows for the code to be introspected by,
        for example, `pdb` or `inspect`"""
        import ast
        import linecache

        if not hasattr(exec_saving_source, "sources"):
            old_getlines = linecache.getlines
            sources: list[str] = []

            def patched_getlines(filename: str, module_globals: Any = None):
                if "<exec#" in filename:
                    index = int(filename.split("#")[1].split(">")[0])
                    return sources[index].splitlines(True)
                else:
                    return old_getlines(filename, module_globals)

            linecache.getlines = patched_getlines
            setattr(exec_saving_source, "sources", sources)
        sources: list[str] = getattr(exec_saving_source, "sources")
        sources.append(source)
        exec(
            compile(
                ast.parse(source), filename=f"<exec#{len(sources) - 1}>", mode="exec"
            ),
            globals,
            locals,
        )

else:
    exec_saving_source = exec
