"""Independent checks for HW4 interpolation, discounting, and cashflow risk.

Run from the repository root with ``python -m unittest discover -s tests``.
Finite differences appear only in these verification tests; the assignment's
reported sensitivities use analytic integrated-interpolation weights.
"""

import calendar
import datetime as dt
import math
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from fixedincomelib import (
    qfAccrued,
    qfAddPeriod,
    qfCreateBuildMethod,
    qfCreateData1D,
    qfCreateDataCollection,
    qfCreateModel,
    qfCreateModelBuildMethodCollection,
    qfCreateProductFixedAccruedCashflow,
    qfDiscountFactor,
)
from fixedincomelib.date import Date
from fixedincomelib.utilities.numerics import (
    ExtrapMethod, Interpolator1DPCP, InterpMethod,
)
from fixedincomelib.yield_curve.yield_curve_model import (
    YieldCurve, YieldCurveModelComponent,
)


VALUE_DATE = "2026-10-07"
RATES = np.arange(30, 40, dtype=float) / 1000
# Following/USGS dates independently checked against the calendar. In 2028,
# 2029, 2034 and 2035 the weekend is followed by the October federal holiday.
NODE_DATES = [
    "2027-10-07", "2028-10-10", "2029-10-09", "2030-10-07", "2031-10-07",
    "2032-10-07", "2033-10-07", "2034-10-10", "2035-10-09", "2036-10-07",
]


def manual_isda(start, end):
    """ACT/ACT ISDA using Python dates, independent of the library day counter."""
    start, end = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    total = 0.0
    while start < end:
        stop = min(end, dt.date(start.year + 1, 1, 1))
        total += (stop - start).days / (366 if calendar.isleap(start.year) else 365)
        start = stop
    return total


def make_model(projection_rates=RATES, funding_spread=0.0):
    projection_data = qfCreateData1D(
        "INSTANTANEOUS FORWARD RATE", "SOFR-1B-IFR",
        pd.DataFrame({"values": projection_rates}, index=[f"{i}Y" for i in range(1, 11)]),
    )
    funding_data = qfCreateData1D(
        "INSTANTANEOUS FORWARD RATE", "SOFR-1B-FUNDING-IFR",
        pd.DataFrame({"values": [funding_spread]}, index=["1Y"]),
    )
    projection_bm = qfCreateBuildMethod("YIELD_CURVE_INDEX", {
        "TARGET": "SOFR-1B", "INSTANTANEOUS FORWARD RATE": "SOFR-1B-IFR",
    })
    funding_bm = qfCreateBuildMethod("YIELD_CURVE_FUNDING", {
        "TARGET": "SOFR-1B-FLAT", "REFERENCE INDEX": "SOFR-1B",
        "INSTANTANEOUS FORWARD RATE": "SOFR-1B-FUNDING-IFR",
    })
    model = qfCreateModel(
        VALUE_DATE, "YIELD_CURVE",
        qfCreateDataCollection([projection_data, funding_data]),
        qfCreateModelBuildMethodCollection([funding_bm, projection_bm]),
    )
    return (model, model.retrieve_model_component(projection_bm.target_index),
            model.retrieve_model_component(funding_bm.target_index))


class PCPInterpolationTests(unittest.TestCase):
    def setUp(self):
        self.pcp = Interpolator1DPCP(
            np.array([1.0, 3.0, 5.0, 7.0]),
            np.array([3.0, 4.0, 5.0, 6.0]), ExtrapMethod.FLAT,
        )

    def test_left_continuity_and_flat_wings_with_point_gradients(self):
        for x, index in [(-9, 0), (1, 0), (1.01, 1), (3, 1), (3.01, 2),
                         (5, 2), (5.01, 3), (7, 3), (99, 3)]:
            with self.subTest(x=x):
                self.assertEqual(self.pcp.interpolate(x), [3, 4, 5, 6][index])
                np.testing.assert_array_equal(
                    self.pcp.gradient_wrt_ordinate(x), np.eye(4)[index],
                )

    def test_oriented_integrals_and_hand_calculated_exposure_lengths(self):
        # Each expected entry is the length spent on that ordinate's support.
        for start, end, weights in [
            (0.5, 6, [0.5, 2, 2, 1]),
            (-1, 9, [2, 2, 2, 4]),
            (1, 3, [0, 2, 0, 0]),
            (6, 9, [0, 0, 0, 3]),
            (-3, 0, [3, 0, 0, 0]),
        ]:
            weights = np.array(weights, dtype=float)
            with self.subTest(start=start, end=end):
                np.testing.assert_allclose(
                    self.pcp.gradient_of_integrated_value_wrt_ordinate(start, end), weights,
                    rtol=0, atol=1e-14,
                )
                self.assertAlmostEqual(self.pcp.integrate(start, end), weights @ [3, 4, 5, 6])
                np.testing.assert_allclose(
                    self.pcp.gradient_of_integrated_value_wrt_ordinate(end, start), -weights,
                    rtol=0, atol=1e-14,
                )
                self.assertAlmostEqual(self.pcp.integrate(end, start), -(weights @ [3, 4, 5, 6]))

    def test_zero_length_and_single_node(self):
        for x in [-2, 1, 4, 7, 10]:
            self.assertEqual(self.pcp.integrate(x, x), 0)
            np.testing.assert_array_equal(
                self.pcp.gradient_of_integrated_value_wrt_ordinate(x, x), np.zeros(4),
            )
        pcp = Interpolator1DPCP(np.array([1.0]), np.array([-0.002]), ExtrapMethod.FLAT)
        self.assertEqual(pcp.interpolate(-5), -0.002)
        self.assertEqual(pcp.interpolate(10), -0.002)
        self.assertAlmostEqual(pcp.integrate(-2, 9), -0.022)
        np.testing.assert_array_equal(pcp.gradient_wrt_ordinate(10), [1.0])
        np.testing.assert_array_equal(pcp.gradient_of_integrated_value_wrt_ordinate(-2, 9), [11.0])


class DiscountFactorTests(unittest.TestCase):
    def test_model_state_dates_and_curve_clock(self):
        _, projection, _ = make_model()
        expected_times = [manual_isda(VALUE_DATE, end) for end in NODE_DATES]
        np.testing.assert_allclose(projection.state_data[0], expected_times, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(projection.state_data[1], RATES)
        self.assertEqual(qfAddPeriod(VALUE_DATE, "2Y", "F", "USGS"), "2028-10-10")

    def test_value_date_and_flat_rate_with_nonzero_funding(self):
        end = "2031-10-07"
        time = manual_isda(VALUE_DATE, end)
        for spread in [0, 0.004, -0.002]:
            with self.subTest(spread=spread):
                model, projection, funding = make_model(np.full(10, 0.03), spread)
                for curve in ["SOFR-1B", "SOFR-1B-FLAT"]:
                    self.assertEqual(qfDiscountFactor(model, curve, VALUE_DATE), 1.0)
                self.assertAlmostEqual(projection.discount_factor(Date(end)), math.exp(-0.03 * time))
                self.assertAlmostEqual(funding.discount_factor(Date(end)), math.exp(-spread * time))
                self.assertAlmostEqual(qfDiscountFactor(model, "SOFR-1B-FLAT", end),
                                       math.exp(-(0.03 + spread) * time))

    def test_flat_extrapolation_beyond_last_node(self):
        model, _, _ = make_model(funding_spread=0.001)
        end = "2041-10-07"
        time = manual_isda(VALUE_DATE, end)
        node_times = np.array([manual_isda(VALUE_DATE, end) for end in NODE_DATES])
        # The final ordinate starts immediately after the penultimate knot and
        # continues flat beyond the final knot.
        weights = np.diff(np.r_[0, node_times[:-1], time])
        expected = math.exp(-(weights @ RATES + 0.001 * time))
        self.assertAlmostEqual(qfDiscountFactor(model, "SOFR-1B-FLAT", end), expected)

    def test_invalid_discount_dates(self):
        model, projection, _ = make_model()
        for end in ["", "2026-10-06"]:
            with self.subTest(end=end), self.assertRaises(ValueError):
                projection.discount_factor(Date(end))
            with self.subTest(end=end, full_curve=True), self.assertRaises(ValueError):
                qfDiscountFactor(model, "SOFR-1B-FLAT", end)

    def test_multiple_references_use_stored_identifiers_and_detect_cycles(self):
        class Identifier:
            def __init__(self, name, reference=None):
                self.name_ = name
                self.reference_index = reference

            def name(self):
                return self.name_

        base = Identifier("BASE")
        middle = Identifier("MIDDLE", base)
        outer = Identifier("OUTER", middle)
        model = YieldCurve(Date(VALUE_DATE), None, None)
        bm = SimpleNamespace(
            interpolation_method=InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
            extrapolation_method=ExtrapMethod.FLAT,
        )
        for index, rate in [(base, 0.03), (middle, 0.004), (outer, -0.001)]:
            model.set_model_component(index.name(), YieldCurveModelComponent(
                Date(VALUE_DATE), index, np.array([[1.0], [rate]]), bm,
            ))
        end = "2031-10-07"
        # Querying with a fresh identifier still follows the stored reference.
        self.assertAlmostEqual(
            model.discount_factor(Identifier("OUTER"), Date(end)),
            math.exp(-0.033 * manual_isda(VALUE_DATE, end)),
        )
        base.reference_index = outer
        with self.assertRaisesRegex(ValueError, "Circular"):
            model.discount_factor(outer, Date(end))


class CashflowAnalyticRiskTests(unittest.TestCase):
    def test_five_year_price_and_all_analytic_node_sensitivities(self):
        model, projection, funding = make_model()
        end = qfAddPeriod(VALUE_DATE, "5Y", "F", "USGS")
        product = qfCreateProductFixedAccruedCashflow(
            VALUE_DATE, end, "USD", 100, "ACT/360", payment_date=end,
        )
        self.assertEqual(end, "2031-10-07")
        cash = product.notional * product.accrued
        expected_cash = 100 * (dt.date(2031, 10, 7) - dt.date(2026, 10, 7)).days / 360
        self.assertAlmostEqual(cash, expected_cash)
        time = qfAccrued(VALUE_DATE, end, "ACT/ACT (ISDA)")
        self.assertAlmostEqual(time, manual_isda(VALUE_DATE, end))
        self.assertNotAlmostEqual(time, product.accrued)

        first_five_times = [manual_isda(VALUE_DATE, end) for end in NODE_DATES[:5]]
        expected_projection_weights = np.r_[np.diff([0] + first_five_times), np.zeros(5)]
        projection_weights = projection.state_data_interpolator.gradient_of_integrated_value_wrt_ordinate(0, time)
        funding_weights = funding.state_data_interpolator.gradient_of_integrated_value_wrt_ordinate(0, time)
        np.testing.assert_allclose(projection_weights, expected_projection_weights, rtol=0, atol=1e-14)
        np.testing.assert_allclose(funding_weights, [time], rtol=0, atol=1e-14)

        price = cash * qfDiscountFactor(model, "SOFR-1B-FLAT", end)
        expected_price = expected_cash * math.exp(-(expected_projection_weights @ RATES))
        self.assertAlmostEqual(price, expected_price)
        risk = -price * np.r_[projection_weights, funding_weights]
        expected_risk = -expected_price * np.r_[expected_projection_weights, time]
        np.testing.assert_allclose(risk, expected_risk, rtol=0, atol=1e-10)
        self.assertEqual(len(risk), 11)
        self.assertEqual(funding.state_data[1][0], 0)
        self.assertLess(risk[-1], 0)  # A zero funding spread still has risk.
        np.testing.assert_array_equal(risk[5:10], np.zeros(5))

        # Independent central differences verify the complete priced cashflow,
        # including rebuilding interpolators after every parameter change.
        bump = 1e-6
        numerical_risk = []
        for component in [projection, funding]:
            for i in range(component.num_state_data):
                original = component.state_data[1][i]
                try:
                    component.perturb_model_parameter(i, original + bump, True)
                    up = cash * qfDiscountFactor(model, "SOFR-1B-FLAT", end)
                    component.perturb_model_parameter(i, original - bump, True)
                    down = cash * qfDiscountFactor(model, "SOFR-1B-FLAT", end)
                    numerical_risk.append((up - down) / (2 * bump))
                finally:
                    component.perturb_model_parameter(i, original, True)
        np.testing.assert_allclose(risk, numerical_risk, rtol=1e-8, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
