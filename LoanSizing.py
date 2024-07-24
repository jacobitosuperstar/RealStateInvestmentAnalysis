"""
Loan Sizing
"""
from typing import Union, Literal, Iterable, Dict, Any, Tuple
import math
# from dataclasses import dataclass

import numpy as np
import numpy_financial as npf


class LoanMetrics:
    LTV: np.float64 = np.float64(65 / 100) # percentage
    "`LTV: ` Maximum loan to value ratio allowed"
    LTC: np.float64 = np.float64(70 / 100) # percentage
    "`LTC: ` Maximum loan to cost ratio allowed"
    DSCR: np.float64 = np.float64(1.25) # debt service coverage ratio
    """`DSCR:` Debt service coverage ratio.
    Annual Net Operating Income / Annual Debt Service"""
    DEBT_YIELD: np.float64 = np.float64(7.6 / 100) # percentage
    """`DebtYield: ` Net operating income against the total loan ammount.
    Annual Net Operating Income / Total Loan Ammount"""

    def __init__(self,
        purchase_price: int,
        appraised_value: int,
        closing_and_renovations: int,
        interest_rate: float,
        net_operating_income: int,
        amortization: int,
        term: int,
        payment_type: Literal["begin", "end"] = "end",
    ) -> None:
        """Class representation of the Loan Metrics that come from a commercial
        real state that is going to be bougth by a client.

        Parameters
        ----------
        purchase_price: int
            Value on which the property is being bougth.
        appraised_value: int
            Value on which the property is appraised.
        closing_and_renovations: int
            Value of closing the deal and renovate the property if nescessary
        interest_rate: Decimal
            Yearly interest rate.
        net_operating_income: int
            Generated expenses minus operation expenses and excluding one time
            costs.
        amortization: int
            Amount of time on which the loan will be spreaded out.
        term: int
            Time after which the loan will be fully paid.
        payment_type: ["begin", "end"]
            When the payment is done. At the end or the beginning of the month.
            By default is the end of the month.
        """
        self.purchase_price: np.int64 = np.floor(purchase_price).astype(int)
        self.appraised_value: np.int64 = np.floor(appraised_value).astype(int)
        self.closing_and_renovations: int = np.floor(closing_and_renovations).astype(int)
        self.interest_rate: np.float64 = np.float64(interest_rate)
        self.net_operating_income: np.int64 = np.floor(net_operating_income).astype(int)
        self.amortization: np.int64 = np.floor(amortization).astype(int)
        self.term: np.int64 = np.floor(term).astype(int)
        self.payment_type: Literal["begin", "end"] = payment_type

    def payment(
        self,
        present_value: int,
        future_value: int = 0,
    ) -> Union[Iterable[int], int]:
        """Present value of an investment (Whats worth the total ammount of
        payments now)

        Parameters
        ----------
        present_value: int | float
            Ammount borrowed.
        future_value: int | float
            Money owed after the last payment. By default is 0.

        Returns
        -------
        NDArray[float64] | float64
        """

        if not self.payment_type in ["begin", "end"]:
            raise ValueError("Incorrect Value entered for the payment_type parameter")

        calculated_payment = npf.pmt(
            pv=present_value,
            fv=future_value,
            rate=self.interest_rate/12,
            nper=self.amortization * 12,
            when=self.payment_type,
        )
        return np.floor(calculated_payment).astype(int)

    def debt_payments(self) -> float:
        return 0.0

    def MLA_LTV(self,) -> Union[float, int]:
        """Maximum loan to value. This tells us how much value of the appraised
        property can be lended to the client.

        Returns
        -------
        Decimal
        """
        MLA_LTV = self.appraised_value * self.LTV
        return np.floor(MLA_LTV).astype(int)

    def MLA_LTC(self,) -> Union[int, float]:
        """Maximum loan to cost. This tells us how much of the total value of the
        deal can be lended to the client.

        Returns
        -------
        Decimal
        """
        MLA_LTC = self.LTC * (self.purchase_price + self.closing_and_renovations)
        return np.floor(MLA_LTC).astype(int)

    def MLA_DSCR(
        self,
        future_value: int = 0,
    ) -> Union[Iterable[int], int]:
        """Maximum loan ammout given the DSCR constrain. For this is the
        present value of an investment (Whats worth the total ammount of
        payments now).

        Parameters
        ----------
        future_value: int | float
            Money owed after the last payment. By default is 0.

        Returns
        -------
        NDArray[float64] | float64
        """
        monthly_rate = self.interest_rate / 12
        nper = self.amortization * 12
        pmt = - self.net_operating_income / self. DSCR/ 12
        present_value = npf.pv(
            rate=monthly_rate,
            nper=nper,
            pmt=pmt,
            fv=future_value,
            when=self.payment_type,
        )
        return np.floor(present_value).astype(int)

    def MLA_DebtYield(self,) -> int:
        MLA_DebtYield = self.net_operating_income / self.DEBT_YIELD
        return np.floor(MLA_DebtYield).astype(int)

    def loan_sizing(
        self,
        future_value: int = 0,
    ) -> Dict[str, Any]:
        """Condensates all the calculations from all the maximum loan ammounts
        into a single dictionary.

        Parameters
        ----------
        future_value: int,
            Remaining capital at the end of the loan.

        Returns
        -------
        Dict
            All the loan sizing metrics.
        """
        sized_loan: Dict[str, Any] = {
            "MLA_LTV": self.MLA_LTV(),
            "MLA_LTC": self.MLA_LTC(),
            "MLA_DSCR": self.MLA_DSCR(future_value=future_value),
            "MLA_DebtYield": self.MLA_DebtYield(),
        }
        return sized_loan

    def mLA(
        self,
        future_value: int = 0,
    ) -> Tuple:
        """Minimum loan ammount

        Parameters
        ----------
        future_value: int | float
            Money owed after the last payment. By default is 0.

        Returns
        -------
        Tuple[str, Union[float, int]]
            Returns the loan type with the minimum value from all the loan
            sizing metrics.
        """
        loan_sizing = self.loan_sizing(future_value=future_value)
        min_value = min(loan_sizing.values())
        mla = ()
        for key, value in loan_sizing.items():
            if value == min_value:
                mla = (key, value)
                break
        return mla

    def capitalization_rate(self):
        """Capitalization (Cap) Rate. Yield on net operating income based on
        the purchase price without taking into account debt or capital expenses
        on the deal.
        """
        cap_rate = self.net_operating_income / self.purchase_price
        return np.round(cap_rate, decimals=5)

def leverage(
    purchase_price: int,
    loan_to_value_ratio: Union[int, float],
    value_change: Union[int, float],
    percentage: bool = False,
) -> Dict:

    if percentage is True:
        loan_to_value_ratio = round(loan_to_value_ratio / 100, 5)
        value_change = round(value_change / 100, 5)

    purchase_price = math.floor(purchase_price)

    loan_ammount = round(purchase_price * loan_to_value_ratio, 5)
    equity_balance =  round(purchase_price - loan_ammount, 5)
    new_equity_balance = round(purchase_price * (1 + value_change) - loan_ammount, 5)
    equity_increase = round(new_equity_balance / equity_balance - 1, 5)
    multiple_on_value_increase = round(equity_increase / value_change, 5)

    new_equity_stats = {
        "purchase_price": purchase_price,
        "loan_ammount": loan_ammount,
        "equity_balance": equity_balance,
        "new_equity_balance": new_equity_balance,
        "equity_increase": equity_increase,
        "multiple_on_value_increase": multiple_on_value_increase,
    }
    return new_equity_stats


if __name__ == "__main__":

    loan_metrics = LoanMetrics(
        purchase_price = 10_000_000,
        appraised_value = 10_000_000,
        closing_and_renovations = 500_000,
        interest_rate = 0.045,
        net_operating_income = 500_000,
        amortization = 25,
        term = 10,
        payment_type = "end",
    )

    loan_sizing = loan_metrics.loan_sizing(0)
    print("loan_sizing: ", loan_sizing)

    minimum_loan_ammount = loan_metrics.mLA(0)[1]
    print("minimum_loan_ammount: ", minimum_loan_ammount)

    loan_payments = loan_metrics.payment(
        present_value=minimum_loan_ammount,
        future_value=0,
    )
    print("loan_payments: ", loan_payments)

    cap_rate = loan_metrics.capitalization_rate()
    print("capitalization_rate: ", cap_rate)

    leveraged_earnings = leverage(
        purchase_price=10_000_000,
        loan_to_value_ratio=80,
        value_change=-20,
        percentage=True,
    )

    print("leveraged_earnings_stats: ", leveraged_earnings)
