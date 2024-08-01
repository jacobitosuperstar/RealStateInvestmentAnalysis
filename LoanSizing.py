"""
Loan Sizing
"""
from typing import Union, Literal, Iterable, Dict, Any, Tuple
import json
import warnings
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
        percentage: bool = True,
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
        self.purchase_price: int = math.floor(purchase_price)
        self.appraised_value: int = np.floor(appraised_value)
        self.closing_and_renovations: int = np.floor(closing_and_renovations)
        self.net_operating_income: int = math.floor(net_operating_income)
        self.amortization: int = math.floor(amortization)
        self.term: int = math.floor(term)
        self.payment_type: Literal["begin", "end"] = payment_type

        if percentage:
            self.interest_rate: float = round(interest_rate / 100, 4)
        else:
            self.interest_rate: float = round(interest_rate, 4)

    def __str__(self) -> str:
        loan_metrics = {
            "LTV Constraint (%)": round(self.LTV * 100, 2),
            "LTC Constraint (%)": round(self.LTC * 100, 2),
            "DSCR Constraint": self.DSCR,
            "DebtYield Constraint (%)": round(self.DEBT_YIELD * 100, 2),
            "PurchasePrice": self.purchase_price,
            "AppraisedValue": self.appraised_value,
            "ClosingAndRenovations": self.closing_and_renovations,
            "InterestRate(%)": round(self.interest_rate * 100, 2),
            "NOI": self.net_operating_income,
            "Amortization": self.amortization,
            "Term": self.term,
            "PaymentType": self.payment_type,
        }
        return json.dumps(loan_metrics, indent=4)

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

        if self.payment_type not in ["begin", "end"]:
            raise ValueError("Incorrect Value entered for the payment_type parameter")

        calculated_payment = npf.pmt(
            pv=present_value,
            fv=future_value,
            rate=self.interest_rate/12,
            nper=self.amortization * 12,
            when=self.payment_type,
        )
        return np.round(calculated_payment, 2)

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
        return math.floor(MLA_LTV)

    def MLA_LTC(self,) -> Union[int, float]:
        """Maximum loan to cost. This tells us how much of the total value of the
        deal can be lended to the client.

        Returns
        -------
        Decimal
        """
        MLA_LTC = self.LTC * (self.purchase_price + self.closing_and_renovations)
        return math.floor(MLA_LTC)

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
        return math.floor(present_value)

    def MLA_DebtYield(self,) -> int:
        MLA_DebtYield = self.net_operating_income / self.DEBT_YIELD
        return math.floor(MLA_DebtYield)

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


def GradientDescentReqNOIMaxLoan(
    initial_noi: int,
    initial_lm: LoanMetrics,
    future_value: int = 0,
    max_iterations: int = 1_000_000,
    learning_rate = 100_000,
    tolerance: float = 0.005,
    ) -> Union[int, float]:
    """Returns the maximum net operating income needed to get the maximum loan
    ammount and still meet the loan constrains.

    The maximum loan ammount will always be the maximum Loan To Value ratio
    allowed by the bank.
    """

    net_operating_income = initial_noi
    iterations = 0

    while max_iterations > iterations:

        loan_metrics: LoanMetrics = LoanMetrics(
            purchase_price=initial_lm.purchase_price,
            appraised_value=initial_lm.appraised_value,
            closing_and_renovations=initial_lm.closing_and_renovations,
            interest_rate=initial_lm.interest_rate,
            net_operating_income=net_operating_income,
            amortization=initial_lm.amortization,
            term=initial_lm.term,
            payment_type=initial_lm.payment_type,
        )

        min_loan_ammount = loan_metrics.mLA(future_value)[-1]
        max_loan_ammount = loan_metrics.MLA_LTV()

        diff = round((min_loan_ammount/max_loan_ammount) - 1, 2)

        if abs(diff) <= tolerance:
            return net_operating_income

        net_operating_income -= learning_rate * diff

        if net_operating_income <= 0:
            raise ValueError("The Net Operating Income is 0 or bellow.")

        if math.isnan(net_operating_income):
            net_operating_income: float = round(0, 4)
            warnings.warn(
                f"The initial Purchase Price was too high, starting the search from {net_operating_income}",
                RuntimeWarning
            )

        net_operating_income = math.floor(net_operating_income)

        iterations += 1
    else:
        raise ValueError(f"Searched value not found. Last Iteration ended in {net_operating_income}")


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

    net_operating_income = 250_000

    loan_metrics = LoanMetrics(
        purchase_price = 10_000_000,
        appraised_value = 10_000_000,
        closing_and_renovations = 500_000,
        interest_rate = 4.50,
        net_operating_income = net_operating_income,
        amortization = 25,
        term = 10,
        payment_type = "end",
    )
    print(loan_metrics)

    loan_sizing = loan_metrics.loan_sizing(0)
    string = json.dumps(loan_sizing, indent=4)
    print("loan_sizing: ", string)

    minimum_loan_ammount = loan_metrics.mLA(0)[1]
    print("minimum_loan_ammount: ", minimum_loan_ammount)

    loan_payments = loan_metrics.payment(
        present_value=minimum_loan_ammount,
        future_value=0,
    )
    print("loan_payments: ", loan_payments)

    cap_rate = loan_metrics.capitalization_rate()
    print("capitalization_rate: ", cap_rate)

    # leveraged_earnings = leverage(
    #     purchase_price=10_000_000,
    #     loan_to_value_ratio=80,
    #     value_change=-20,
    #     percentage=True,
    # )

    # print("leveraged_earnings_stats: ", leveraged_earnings)

    noi = GradientDescentReqNOIMaxLoan(
        initial_noi=net_operating_income,
        initial_lm=loan_metrics,
        future_value=0,
        max_iterations=100_000,
        learning_rate=100_000,
        tolerance=0.005,
    )
    print("Required Net Operating Income for Loan Ammount: ", noi)
