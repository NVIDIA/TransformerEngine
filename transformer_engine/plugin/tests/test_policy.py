# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Test suite for TE-FL scheduling policy system.

This module tests:
1. SelectionPolicy creation and configuration
2. Environment variable parsing
3. Policy context managers
4. Vendor filtering (allow/deny)
5. Per-operator custom ordering
6. PolicyManager singleton and thread safety
7. Integration with OpManager
"""

import os
import sys
import threading
import unittest
from unittest.mock import patch
from typing import List, Dict


class TestSelectionPolicy(unittest.TestCase):
    """Test SelectionPolicy dataclass and methods"""

    def setUp(self):
        """Import policy module fresh for each test"""
        from transformer_engine.plugin.core.policy import (
            SelectionPolicy,
            PREFER_DEFAULT,
            PREFER_VENDOR,
            PREFER_REFERENCE,
        )
        self.SelectionPolicy = SelectionPolicy
        self.PREFER_DEFAULT = PREFER_DEFAULT
        self.PREFER_VENDOR = PREFER_VENDOR
        self.PREFER_REFERENCE = PREFER_REFERENCE

    def test_default_policy_creation(self):
        """Test creating policy with default values"""
        policy = self.SelectionPolicy.from_dict()

        self.assertEqual(policy.prefer, self.PREFER_DEFAULT)
        self.assertFalse(policy.strict)
        self.assertEqual(policy.per_op_order, ())
        self.assertEqual(policy.deny_vendors, frozenset())
        self.assertIsNone(policy.allow_vendors)
        print("    [PASS] Default policy creation")

    def test_policy_with_prefer_vendor(self):
        """Test creating policy with vendor preference"""
        policy = self.SelectionPolicy.from_dict(prefer="vendor")

        self.assertEqual(policy.prefer, "vendor")
        self.assertEqual(policy.get_default_order(), ["vendor", "flagos", "reference"])
        print("    [PASS] Policy with vendor preference")

    def test_policy_with_prefer_reference(self):
        """Test creating policy with reference preference"""
        policy = self.SelectionPolicy.from_dict(prefer="reference")

        self.assertEqual(policy.prefer, "reference")
        self.assertEqual(policy.get_default_order(), ["reference", "flagos", "vendor"])
        print("    [PASS] Policy with reference preference")

    def test_policy_with_prefer_flagos(self):
        """Test creating policy with flagos preference (default)"""
        policy = self.SelectionPolicy.from_dict(prefer="flagos")

        self.assertEqual(policy.prefer, "flagos")
        self.assertEqual(policy.get_default_order(), ["flagos", "vendor", "reference"])
        print("    [PASS] Policy with flagos preference")

    def test_invalid_prefer_value(self):
        """Test that invalid prefer value raises error"""
        with self.assertRaises(ValueError) as context:
            self.SelectionPolicy.from_dict(prefer="invalid")

        self.assertIn("Invalid prefer value", str(context.exception))
        print("    [PASS] Invalid prefer value raises error")

    def test_strict_mode(self):
        """Test strict mode setting"""
        policy = self.SelectionPolicy.from_dict(strict=True)

        self.assertTrue(policy.strict)
        print("    [PASS] Strict mode setting")

    def test_deny_vendors(self):
        """Test deny vendors configuration"""
        policy = self.SelectionPolicy.from_dict(deny_vendors={"rocm", "dcu"})

        self.assertEqual(policy.deny_vendors, frozenset({"rocm", "dcu"}))
        self.assertFalse(policy.is_vendor_allowed("rocm"))
        self.assertFalse(policy.is_vendor_allowed("dcu"))
        self.assertTrue(policy.is_vendor_allowed("cuda"))
        print("    [PASS] Deny vendors configuration")

    def test_allow_vendors(self):
        """Test allow vendors whitelist"""
        policy = self.SelectionPolicy.from_dict(allow_vendors={"cuda"})

        self.assertEqual(policy.allow_vendors, frozenset({"cuda"}))
        self.assertTrue(policy.is_vendor_allowed("cuda"))
        self.assertFalse(policy.is_vendor_allowed("rocm"))
        print("    [PASS] Allow vendors whitelist")

    def test_deny_overrides_allow(self):
        """Test that deny takes precedence over allow"""
        policy = self.SelectionPolicy.from_dict(
            allow_vendors={"cuda", "rocm"},
            deny_vendors={"rocm"},
        )

        self.assertTrue(policy.is_vendor_allowed("cuda"))
        self.assertFalse(policy.is_vendor_allowed("rocm"))
        print("    [PASS] Deny overrides allow")

    def test_per_op_order(self):
        """Test per-operator custom ordering"""
        policy = self.SelectionPolicy.from_dict(
            per_op_order={
                "layernorm_fwd": ["vendor", "flagos"],
                "rmsnorm_fwd": ["flagos", "reference"],
            }
        )

        self.assertEqual(policy.get_per_op_order("layernorm_fwd"), ["vendor", "flagos"])
        self.assertEqual(policy.get_per_op_order("rmsnorm_fwd"), ["flagos", "reference"])
        self.assertIsNone(policy.get_per_op_order("unknown_op"))
        print("    [PASS] Per-operator custom ordering")

    def test_policy_fingerprint(self):
        """Test policy fingerprint generation"""
        policy1 = self.SelectionPolicy.from_dict(prefer="vendor", strict=True)
        policy2 = self.SelectionPolicy.from_dict(prefer="vendor", strict=True)
        policy3 = self.SelectionPolicy.from_dict(prefer="flagos", strict=True)

        self.assertEqual(policy1.fingerprint(), policy2.fingerprint())
        self.assertNotEqual(policy1.fingerprint(), policy3.fingerprint())
        print("    [PASS] Policy fingerprint generation")

    def test_policy_immutability(self):
        """Test that SelectionPolicy is immutable (frozen dataclass)"""
        policy = self.SelectionPolicy.from_dict(prefer="vendor")

        with self.assertRaises(AttributeError):
            policy.prefer = "flagos"  # Should fail - frozen dataclass
        print("    [PASS] Policy immutability")

    def test_policy_hashable(self):
        """Test that SelectionPolicy is hashable (can be used in sets/dicts)"""
        policy1 = self.SelectionPolicy.from_dict(prefer="vendor")
        policy2 = self.SelectionPolicy.from_dict(prefer="vendor")

        policy_set = {policy1, policy2}
        self.assertEqual(len(policy_set), 1)  # Same policy, should dedupe
        print("    [PASS] Policy hashable")


class TestPolicyManager(unittest.TestCase):
    """Test PolicyManager singleton and state management"""

    def setUp(self):
        """Reset policy manager state before each test"""
        from transformer_engine.plugin.core.policy import (
            PolicyManager,
            reset_global_policy,
        )
        reset_global_policy()
        self.PolicyManager = PolicyManager

    def tearDown(self):
        """Clean up after each test"""
        from transformer_engine.plugin.core.policy import reset_global_policy
        reset_global_policy()
        # Clear any test environment variables
        for key in ["TE_FL_PREFER", "TE_FL_PREFER_VENDOR", "TE_FL_STRICT",
                    "TE_FL_DENY_VENDORS", "TE_FL_ALLOW_VENDORS", "TE_FL_PER_OP"]:
            os.environ.pop(key, None)

    def test_singleton_pattern(self):
        """Test PolicyManager is a singleton"""
        manager1 = self.PolicyManager.get_instance()
        manager2 = self.PolicyManager.get_instance()

        self.assertIs(manager1, manager2)
        print("    [PASS] PolicyManager singleton pattern")

    def test_policy_epoch(self):
        """Test policy epoch tracking"""
        from transformer_engine.plugin.core.policy import (
            get_policy_epoch,
            bump_policy_epoch,
        )

        initial_epoch = get_policy_epoch()
        new_epoch = bump_policy_epoch()

        self.assertEqual(new_epoch, initial_epoch + 1)
        self.assertEqual(get_policy_epoch(), new_epoch)
        print("    [PASS] Policy epoch tracking")

    def test_global_policy_set_and_get(self):
        """Test setting and getting global policy"""
        from transformer_engine.plugin.core.policy import (
            SelectionPolicy,
            set_global_policy,
            get_policy,
        )

        custom_policy = SelectionPolicy.from_dict(prefer="vendor", strict=True)
        old_policy = set_global_policy(custom_policy)

        current = get_policy()
        self.assertEqual(current.prefer, "vendor")
        self.assertTrue(current.strict)
        print("    [PASS] Global policy set and get")

    def test_reset_global_policy(self):
        """Test resetting global policy to env defaults"""
        from transformer_engine.plugin.core.policy import (
            SelectionPolicy,
            set_global_policy,
            reset_global_policy,
            get_policy,
        )

        # Set custom policy
        custom_policy = SelectionPolicy.from_dict(prefer="vendor")
        set_global_policy(custom_policy)

        # Reset to defaults
        reset_global_policy()

        current = get_policy()
        self.assertEqual(current.prefer, "flagos")  # Default
        print("    [PASS] Reset global policy")


class TestEnvironmentVariables(unittest.TestCase):
    """Test environment variable parsing"""

    def setUp(self):
        """Clear environment and reset policy"""
        from transformer_engine.plugin.core.policy import reset_global_policy
        reset_global_policy()
        # Clear all test env vars
        for key in ["TE_FL_PREFER", "TE_FL_PREFER_VENDOR", "TE_FL_STRICT",
                    "TE_FL_DENY_VENDORS", "TE_FL_ALLOW_VENDORS", "TE_FL_PER_OP"]:
            os.environ.pop(key, None)

    def tearDown(self):
        """Clean up environment"""
        for key in ["TE_FL_PREFER", "TE_FL_PREFER_VENDOR", "TE_FL_STRICT",
                    "TE_FL_DENY_VENDORS", "TE_FL_ALLOW_VENDORS", "TE_FL_PER_OP"]:
            os.environ.pop(key, None)
        from transformer_engine.plugin.core.policy import reset_global_policy
        reset_global_policy()

    def test_te_fl_prefer_flagos(self):
        """Test TE_FL_PREFER=flagos"""
        os.environ["TE_FL_PREFER"] = "flagos"

        from transformer_engine.plugin.core.policy import policy_from_env
        policy = policy_from_env()

        self.assertEqual(policy.prefer, "flagos")
        print("    [PASS] TE_FL_PREFER=flagos")

    def test_te_fl_prefer_vendor(self):
        """Test TE_FL_PREFER=vendor"""
        os.environ["TE_FL_PREFER"] = "vendor"

        from transformer_engine.plugin.core.policy import policy_from_env
        policy = policy_from_env()

        self.assertEqual(policy.prefer, "vendor")
        print("    [PASS] TE_FL_PREFER=vendor")

    def test_te_fl_prefer_reference(self):
        """Test TE_FL_PREFER=reference"""
        os.environ["TE_FL_PREFER"] = "reference"

        from transformer_engine.plugin.core.policy import policy_from_env
        policy = policy_from_env()

        self.assertEqual(policy.prefer, "reference")
        print("    [PASS] TE_FL_PREFER=reference")

    def test_te_fl_prefer_vendor_legacy(self):
        """Test legacy TE_FL_PREFER_VENDOR=1"""
        os.environ["TE_FL_PREFER_VENDOR"] = "1"

        from transformer_engine.plugin.core.policy import policy_from_env
        policy = policy_from_env()

        self.assertEqual(policy.prefer, "vendor")
        print("    [PASS] TE_FL_PREFER_VENDOR=1 (legacy)")

    def test_te_fl_prefer_overrides_legacy(self):
        """Test that TE_FL_PREFER takes precedence over TE_FL_PREFER_VENDOR"""
        os.environ["TE_FL_PREFER"] = "reference"
        os.environ["TE_FL_PREFER_VENDOR"] = "1"

        from transformer_engine.plugin.core.policy import policy_from_env
        policy = policy_from_env()

        self.assertEqual(policy.prefer, "reference")  # TE_FL_PREFER wins
        print("    [PASS] TE_FL_PREFER overrides TE_FL_PREFER_VENDOR")

    def test_te_fl_strict(self):
        """Test TE_FL_STRICT=1"""
        os.environ["TE_FL_STRICT"] = "1"

        from transformer_engine.plugin.core.policy import policy_from_env
        policy = policy_from_env()

        self.assertTrue(policy.strict)
        print("    [PASS] TE_FL_STRICT=1")

    def test_te_fl_deny_vendors(self):
        """Test TE_FL_DENY_VENDORS parsing"""
        os.environ["TE_FL_DENY_VENDORS"] = "rocm,dcu,intel"

        from transformer_engine.plugin.core.policy import policy_from_env
        policy = policy_from_env()

        self.assertEqual(policy.deny_vendors, frozenset({"rocm", "dcu", "intel"}))
        print("    [PASS] TE_FL_DENY_VENDORS parsing")

    def test_te_fl_allow_vendors(self):
        """Test TE_FL_ALLOW_VENDORS parsing"""
        os.environ["TE_FL_ALLOW_VENDORS"] = "cuda,rocm"

        from transformer_engine.plugin.core.policy import policy_from_env
        policy = policy_from_env()

        self.assertEqual(policy.allow_vendors, frozenset({"cuda", "rocm"}))
        print("    [PASS] TE_FL_ALLOW_VENDORS parsing")

    def test_te_fl_per_op(self):
        """Test TE_FL_PER_OP parsing"""
        os.environ["TE_FL_PER_OP"] = "layernorm_fwd=vendor|flagos;rmsnorm_fwd=flagos|reference"

        from transformer_engine.plugin.core.policy import policy_from_env
        policy = policy_from_env()

        self.assertEqual(policy.get_per_op_order("layernorm_fwd"), ["vendor", "flagos"])
        self.assertEqual(policy.get_per_op_order("rmsnorm_fwd"), ["flagos", "reference"])
        print("    [PASS] TE_FL_PER_OP parsing")


class TestContextManagers(unittest.TestCase):
    """Test policy context managers"""

    def setUp(self):
        """Reset policy before each test"""
        from transformer_engine.plugin.core.policy import reset_global_policy
        reset_global_policy()

    def tearDown(self):
        """Clean up after test"""
        from transformer_engine.plugin.core.policy import reset_global_policy
        reset_global_policy()

    def test_policy_context(self):
        """Test basic policy_context usage"""
        from transformer_engine.plugin.core.policy import (
            SelectionPolicy,
            policy_context,
            get_policy,
        )

        original = get_policy()
        custom = SelectionPolicy.from_dict(prefer="vendor", strict=True)

        with policy_context(custom):
            inside = get_policy()
            self.assertEqual(inside.prefer, "vendor")
            self.assertTrue(inside.strict)

        after = get_policy()
        self.assertEqual(after.prefer, original.prefer)
        print("    [PASS] policy_context usage")

    def test_with_preference(self):
        """Test with_preference context manager"""
        from transformer_engine.plugin.core.policy import (
            with_preference,
            get_policy,
        )

        original = get_policy()

        with with_preference("vendor"):
            self.assertEqual(get_policy().prefer, "vendor")

        with with_preference("reference"):
            self.assertEqual(get_policy().prefer, "reference")

        self.assertEqual(get_policy().prefer, original.prefer)
        print("    [PASS] with_preference context manager")

    def test_with_strict_mode(self):
        """Test with_strict_mode context manager"""
        from transformer_engine.plugin.core.policy import (
            with_strict_mode,
            get_policy,
        )

        original = get_policy()

        with with_strict_mode():
            self.assertTrue(get_policy().strict)

        self.assertEqual(get_policy().strict, original.strict)
        print("    [PASS] with_strict_mode context manager")

    def test_with_allowed_vendors(self):
        """Test with_allowed_vendors context manager"""
        from transformer_engine.plugin.core.policy import (
            with_allowed_vendors,
            get_policy,
        )

        with with_allowed_vendors("cuda", "rocm"):
            policy = get_policy()
            self.assertEqual(policy.allow_vendors, frozenset({"cuda", "rocm"}))

        self.assertIsNone(get_policy().allow_vendors)
        print("    [PASS] with_allowed_vendors context manager")

    def test_with_denied_vendors(self):
        """Test with_denied_vendors context manager"""
        from transformer_engine.plugin.core.policy import (
            with_denied_vendors,
            get_policy,
        )

        with with_denied_vendors("rocm", "dcu"):
            policy = get_policy()
            self.assertIn("rocm", policy.deny_vendors)
            self.assertIn("dcu", policy.deny_vendors)

        self.assertEqual(get_policy().deny_vendors, frozenset())
        print("    [PASS] with_denied_vendors context manager")

    def test_nested_contexts(self):
        """Test nested context managers"""
        from transformer_engine.plugin.core.policy import (
            with_preference,
            with_strict_mode,
            get_policy,
        )

        with with_preference("vendor"):
            self.assertEqual(get_policy().prefer, "vendor")

            with with_strict_mode():
                policy = get_policy()
                # Note: with_strict_mode creates new policy with current prefer
                self.assertTrue(policy.strict)

            # Back to vendor preference, not strict
            self.assertEqual(get_policy().prefer, "vendor")

        # Back to default
        self.assertEqual(get_policy().prefer, "flagos")
        print("    [PASS] Nested context managers")


class TestTokenMatching(unittest.TestCase):
    """Test token matching for implementation selection"""

    def test_match_flagos_token(self):
        """Test matching 'flagos' token"""
        from transformer_engine.plugin.core.types import OpImpl, BackendImplKind, match_token

        impl = OpImpl(
            op_name="test_op",
            impl_id="test.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=lambda: None,
        )

        self.assertTrue(match_token(impl, "flagos"))
        self.assertFalse(match_token(impl, "vendor"))
        self.assertFalse(match_token(impl, "reference"))
        print("    [PASS] Match flagos token")

    def test_match_vendor_token(self):
        """Test matching 'vendor' token"""
        from transformer_engine.plugin.core.types import OpImpl, BackendImplKind, match_token

        impl = OpImpl(
            op_name="test_op",
            impl_id="test.cuda",
            kind=BackendImplKind.VENDOR,
            fn=lambda: None,
            vendor="cuda",
        )

        self.assertTrue(match_token(impl, "vendor"))
        self.assertFalse(match_token(impl, "flagos"))
        print("    [PASS] Match vendor token")

    def test_match_specific_vendor_token(self):
        """Test matching 'vendor:<name>' token"""
        from transformer_engine.plugin.core.types import OpImpl, BackendImplKind, match_token

        impl = OpImpl(
            op_name="test_op",
            impl_id="test.cuda",
            kind=BackendImplKind.VENDOR,
            fn=lambda: None,
            vendor="cuda",
        )

        self.assertTrue(match_token(impl, "vendor:cuda"))
        self.assertFalse(match_token(impl, "vendor:rocm"))
        print("    [PASS] Match specific vendor token")

    def test_match_impl_token(self):
        """Test matching 'impl:<id>' token"""
        from transformer_engine.plugin.core.types import OpImpl, BackendImplKind, match_token

        impl = OpImpl(
            op_name="test_op",
            impl_id="layernorm_cuda_v2",
            kind=BackendImplKind.VENDOR,
            fn=lambda: None,
            vendor="cuda",
        )

        self.assertTrue(match_token(impl, "impl:layernorm_cuda_v2"))
        self.assertFalse(match_token(impl, "impl:other_impl"))
        print("    [PASS] Match impl token")

    def test_match_reference_token(self):
        """Test matching 'reference' token"""
        from transformer_engine.plugin.core.types import OpImpl, BackendImplKind, match_token

        impl = OpImpl(
            op_name="test_op",
            impl_id="test.reference",
            kind=BackendImplKind.REFERENCE,
            fn=lambda: None,
        )

        self.assertTrue(match_token(impl, "reference"))
        self.assertFalse(match_token(impl, "flagos"))
        self.assertFalse(match_token(impl, "vendor"))
        print("    [PASS] Match reference token")


class TestThreadSafety(unittest.TestCase):
    """Test thread safety of PolicyManager"""

    def test_concurrent_policy_access(self):
        """Test concurrent access to policy"""
        from transformer_engine.plugin.core.policy import (
            SelectionPolicy,
            set_global_policy,
            get_policy,
            reset_global_policy,
        )

        reset_global_policy()
        errors = []
        results = []

        def worker(prefer_value: str, worker_id: int):
            try:
                for _ in range(100):
                    policy = SelectionPolicy.from_dict(prefer=prefer_value)
                    set_global_policy(policy)
                    current = get_policy()
                    # Policy should be one of the valid values
                    if current.prefer not in ["flagos", "vendor", "reference"]:
                        errors.append(f"Worker {worker_id}: Invalid prefer value {current.prefer}")
                results.append(worker_id)
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")

        threads = [
            threading.Thread(target=worker, args=("flagos", 0)),
            threading.Thread(target=worker, args=("vendor", 1)),
            threading.Thread(target=worker, args=("reference", 2)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        self.assertEqual(len(results), 3)
        print("    [PASS] Concurrent policy access")

    def test_policy_epoch_increment(self):
        """Test that policy epoch increments correctly under contention"""
        from transformer_engine.plugin.core.policy import (
            get_policy_epoch,
            bump_policy_epoch,
        )

        initial_epoch = get_policy_epoch()
        increments = 100
        threads_count = 4

        def bump_epochs():
            for _ in range(increments):
                bump_policy_epoch()

        threads = [threading.Thread(target=bump_epochs) for _ in range(threads_count)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        final_epoch = get_policy_epoch()
        expected = initial_epoch + (increments * threads_count)

        self.assertEqual(final_epoch, expected)
        print("    [PASS] Policy epoch increment under contention")


class TestDefaultOrder(unittest.TestCase):
    """Test default selection order based on preference"""

    def test_flagos_preference_order(self):
        """Test selection order with flagos preference"""
        from transformer_engine.plugin.core.policy import SelectionPolicy

        policy = SelectionPolicy.from_dict(prefer="flagos")
        order = policy.get_default_order()

        self.assertEqual(order, ["flagos", "vendor", "reference"])
        print("    [PASS] Flagos preference order")

    def test_vendor_preference_order(self):
        """Test selection order with vendor preference"""
        from transformer_engine.plugin.core.policy import SelectionPolicy

        policy = SelectionPolicy.from_dict(prefer="vendor")
        order = policy.get_default_order()

        self.assertEqual(order, ["vendor", "flagos", "reference"])
        print("    [PASS] Vendor preference order")

    def test_reference_preference_order(self):
        """Test selection order with reference preference"""
        from transformer_engine.plugin.core.policy import SelectionPolicy

        policy = SelectionPolicy.from_dict(prefer="reference")
        order = policy.get_default_order()

        self.assertEqual(order, ["reference", "flagos", "vendor"])
        print("    [PASS] Reference preference order")


def run_all_tests():
    """Run all policy tests"""
    print("\n" + "=" * 60)
    print("TE-FL Scheduling Policy Test Suite")
    print("=" * 60)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestSelectionPolicy,
        TestPolicyManager,
        TestEnvironmentVariables,
        TestContextManagers,
        TestTokenMatching,
        TestThreadSafety,
        TestDefaultOrder,
    ]

    for test_class in test_classes:
        print(f"\n[Testing {test_class.__name__}]")
        tests = loader.loadTestsFromTestCase(test_class)
        for test in tests:
            result = unittest.TestResult()
            test.run(result)
            if result.wasSuccessful():
                pass  # Print statements are in individual tests
            else:
                for failure in result.failures + result.errors:
                    print(f"    [FAIL] {test}: {failure[1]}")
        suite.addTests(tests)

    # Run the full suite for final summary
    print("\n" + "=" * 60)
    print("Final Summary")
    print("=" * 60)

    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)

    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total - failures - errors

    print(f"\nTotal: {total}, Passed: {passed}, Failed: {failures}, Errors: {errors}")

    return failures == 0 and errors == 0


def main():
    """Main entry point"""
    success = run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
