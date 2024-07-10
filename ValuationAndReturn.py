"""
CRE Valuation and Return Of Investment Metrics
"""
from typing import (
    Union,
    Optional,
    Literal,
    Iterable,
    List,
    Dict,
    Any,
    Tuple,
)
import math
import statistics

import numpy as np
import numpy_financial as npf


class DealMetrics:
    """All the information that comes from the client.
    """
    def __init__(self,
        purchase_price: int,
        going_in_cap_rate: float,
        closing_and_renovations: int,
        net_leasable_area: Union[int, float],
        yearly_sf_operating_revenue: Union[int, float],
        yearly_sf_operating_expenses: Union[int, float],
        yearly_sf_capital_reserves: Union[int, float],
        annual_revenue_growth: float,
        annual_expense_growth: float,
        annual_capital_reserve_growth: float,
        percentage: bool = True,
    ) -> None:
        self.purchase_price: int = math.floor(purchase_price)
        self.closing_and_renovations: int = math.floor(closing_and_renovations)
        self.net_leasable_area: Union[int, float] = round(net_leasable_area, 2)

        if percentage:
            self.going_in_cap_rate: float = round(going_in_cap_rate / 100, 4)
            self.annual_revenue_growth: float = round(annual_revenue_growth / 100, 4)
            self.annual_expense_growth: float = round(annual_expense_growth / 100, 4)
            self.annual_capital_reserve_growth: float = round(annual_capital_reserve_growth / 100, 4)
        else:
            self.going_in_cap_rate: float = round(going_in_cap_rate, 4)
            self.annual_revenue_growth: float = round(annual_revenue_growth, 4)
            self.annual_expense_growth: float = round(annual_expense_growth, 4)
            self.annual_capital_reserve_growth: float = round(annual_capital_reserve_growth, 4)

        # Calculated Fields
        self.initial_revenue: Union[int, float] = round(self.net_leasable_area * yearly_sf_operating_revenue, 2)
        self.initial_expenses: Union[int, float] = - round(self.net_leasable_area * yearly_sf_operating_expenses, 2)
        self.initial_capital_reserve: Union[int, float] = - round(self.net_leasable_area * yearly_sf_capital_reserves, 2)

    def __str__(self) -> str:
        """JSON string like that represents the initial metricss of the deal.
        """
        deal_metrics: Dict = {
            "PurchasePrice": self.purchase_price,
            "GoingInCapRate(%)": round(self.going_in_cap_rate * 100, 2),
            "ClosingAndRenovations": self.closing_and_renovations,
            "NetLeasableArea": self.net_leasable_area,
            "Year1Revenue": self.initial_revenue,
            "Year1Expenses": self.initial_expenses,
            "Year1CapitalReserve": self.initial_capital_reserve,
            "AnnualRevenueGrowth(%)": round(self.annual_revenue_growth * 100,2),
            "AnnualExpensesGrowth(%)": round(self.annual_expense_growth * 100, 2),
            "AnnualCapitalReservesGrowth(%)": round(self.annual_capital_reserve_growth * 100, 2),
        }
        return json.dumps(deal_metrics, indent=4)


class LoanTerms:
    """All the information that comes from the bank.
    """
    def __init__(self,
        purchase_price: int,
        loan_to_value_ratio: float,
        loan_origination_fees: float,
        interest_rate: float,
        amortization: int,
        term: int,
        future_value: int = 0,
        percentage: bool = True,
    ) -> None:

        self.amortization: int = math.floor(amortization)
        self.term: int = math.floor(term)

        if percentage:
            self.loan_to_value_ratio: float = round(loan_to_value_ratio / 100, 4)
            self.loan_origination_fees: float = round(loan_origination_fees / 100, 4)
            self.interest_rate: float = round(interest_rate / 100, 4)
        else:
            self.loan_to_value_ratio: float = round(loan_to_value_ratio, 4)
            self.loan_origination_fees: float = round(loan_origination_fees, 4)
            self.interest_rate: float = round(interest_rate, 4)

        # CALCULATED FIELDS
        self.loan_ammount: int = round(purchase_price * self.loan_to_value_ratio, 2)
        self.balloon_payment: float = self.BalloonPayment(future_value=future_value)
        self.yearly_loan_payment: float = self.YearlyLoanPayment(future_value=future_value)


    def __str__(self) -> str:
        loan_terms: Dict = {
            "LoanAmmount": self.loan_ammount,
            "LTV(%)": round(self.loan_to_value_ratio * 100, 2),
            "LoanOriginationFees(%)": round(self.loan_origination_fees * 100, 2),
            "InterestRate(%)": round(self.interest_rate * 100, 2),
            "Amortization": self.amortization,
            "Term": self.term,
            "BalloonPaymen": self.balloon_payment,
            "YearlyLoanPayment": self.yearly_loan_payment,
        }
        return json.dumps(loan_terms, indent=4)

    def MonthlyLoanPayment(
        self,
        future_value: int = 0,
    ) -> float:
        """Monthly payment of the loan. Includes Principal Payment and
        Interest.

        Parameters
        ----------
        future_value: int | float
            Money owed after the last payment. By default is 0.

        Returns
        -------
        float
        """

        if self.loan_ammount == 0:
            return 0.00

        calculated_payment = npf.pmt(
            pv=self.loan_ammount,
            fv=future_value,
            rate=self.interest_rate / 12,
            nper=self.amortization * 12,
        )

        if isinstance(calculated_payment, np.ndarray):
            raise TypeError("This is supposed to return a singular value.")

        return round(calculated_payment, 2)

    def MonthlyLoanInterestPayment(
        self,
        future_value: int = 0,
    ) -> Iterable[float]:
        """Interest paid monthly to the bank for the term of the loan.

        Parameters
        ----------
        future_value: int | float
            Money owed after the last payment. By default is 0.

        Returns
        -------
        Iterable[float]
        """
        if self.loan_ammount == 0:
            return [0.00 for _ in range(self.amortization * 12)]

        interest_payment = npf.ipmt(
            rate=self.interest_rate / 12,
            nper= self.amortization * 12,
            per=np.arange(self.amortization * 12) + 1,
            pv=self.loan_ammount,
            fv=future_value,
        )

        interest_payment = np.round(interest_payment, 2)
        return interest_payment

    def YearlyLoanInterestPayment(
        self,
        future_value: int = 0,
    ) -> List[float]:
        """Interest paid yearly to the bank for the term of the loan.

        Parameters
        ----------
        future_value: int | float
            Money owed after the last payment. By default is 0.

        Returns
        -------
        List[float]
        """
        interest_payment = self.MonthlyLoanInterestPayment(
            future_value=future_value
        )

        yearly_interest_payments: List[float] = []
        for i in range(self.term):
            year = round(sum(interest_payment[i*12 : (i+1)*12]), 2)
            yearly_interest_payments.append(year)
        return yearly_interest_payments

    def YearlyLoanPayment(
        self,
        future_value: int = 0,
    ) -> float:
        """Yearly payment of the loan

        Parameters
        ----------
        future_value: int | float
            Money owed after the last payment. By default is 0.

        Returns
        -------
        int
        """
        yearly_payment = self.MonthlyLoanPayment(future_value=future_value) * 12
        return round(yearly_payment, 2)

    def BalloonPayment(
        self,
        future_value: int = 0,
    ) -> float:
        """Balloon payment of the loan at the end of the term.

        Returns
        -------
        float
        """
        yearly_interest_payments = self.YearlyLoanInterestPayment(future_value=future_value)
        yearly_payment = self.YearlyLoanPayment(future_value=future_value)
        yearly_principal_payment = [
            round(yearly_payment - i, 2) for i in yearly_interest_payments
        ]
        paid_to_the_bank: float = sum(yearly_principal_payment)
        balloon_payment: float = round(self.loan_ammount + paid_to_the_bank, 2)
        return - balloon_payment


class SaleMetrics:
    """All the information regarding the sale at the end of the term. For the
    case of the excersice try to match always the year of sale with the term
    of the loan.
    """
    def __init__(
        self,
        exit_cap_rate: float,
        cost_of_sale: float,
        sale_year:int,
        percentage: bool =True,
    ) -> None:
        self.sale_year: int = sale_year

        if percentage:
            self.exit_cap_rate: float = round(exit_cap_rate / 100, 4)
            self.cost_of_sale: float = round(cost_of_sale / 100, 4)
        else:
            self.exit_cap_rate: float = round(exit_cap_rate, 4)
            self.cost_of_sale: float = round(cost_of_sale, 4)

    def __str__(self) -> str:
        sale_metrics: Dict = {
            "ExitCAPRate(%)": round(self.exit_cap_rate * 100, 2),
            "CostOfSale(%)": round(self.cost_of_sale *100, 2),
            "SaleYear": self.sale_year,
        }
        return json.dumps(sale_metrics, indent=4)

    def ProjectedSalePrice(
        self,
        net_operating_income: Union[int, float]
    ) -> float:
        """This is the sale price given the operating income of the property.
        If I am selling this property at the end of the year 10, I have the use
        the net_operating_income projected on the year 11.

        Returns
        -------
        float
        """
        sale_price: float = net_operating_income / self.exit_cap_rate
        sale_price: float = sale_price - (sale_price * self.cost_of_sale)
        return round(sale_price, 2)


class ReturnOfInvestmentMetrics:
    """How much money I am getting out of this?.

    NOTES: Because the selling of the property is part of the IRR and part of
    the EquityMultiple, we will asume that always the sell of the property is
    whats comming at the end of the term. The value is needed anyway for
    refinancing porpuses too. Also for this case, the year of the sale will
    always be the end of the term.
    """
    def __init__(
        self,
        deal_metrics: DealMetrics,
        loan_terms: LoanTerms,
        sale_metrics: SaleMetrics,
    ) -> None:
        self.deal_metrics: DealMetrics = deal_metrics
        self.loan_terms: LoanTerms = loan_terms
        self.sale_metrics: SaleMetrics  = sale_metrics

        # CALCULATED FIELDS
        self.adquisition_cost: float = self.AdquisitionCost()
        self.yearly_net_cash_flow_projection = self.YearlyNetCashFlowProjection()
        self.levered_net_cash_flows = self.LeveredNetCashFlows()

        # RETURN METRICS
        self.irr: float = self.IRR()
        self.equity_multiple: float = self.EquityMultiple()
        self.average_cash_on_cash_return: float = self.AverageCashOnCashReturn()

    def __str__(self) -> str:
        string = {
            "IRR(%)": self.irr,
            "EquityMultiple": self.equity_multiple,
            "AverageCashOnCashReturn": self.average_cash_on_cash_return,
            "YearlyNetCashFlowProjection": self.yearly_net_cash_flow_projection,
        }
        return json.dumps(string, indent=4)

    def AdquisitionCost(self) -> float:
        """The money that I have to put for the purchase of the CRE.

        Returns
        -------
        float
        """
        net_cash_flow = (
            - self.deal_metrics.purchase_price +
            - self.deal_metrics.closing_and_renovations +
            - (self.loan_terms.loan_origination_fees * self.loan_terms.loan_ammount) +
            self.loan_terms.loan_ammount
        )
        return round(net_cash_flow, 2)

    def CashOnCashReturn(
        self,
        net_cash_flow: Union[int, float],
    ) -> float:
        """Cash on Cash return. Return made in reference to the money invested
        to adquire the property.

        Returns
        -------
        float
            Percentage that represents the CashOnCashReturn of investment.
        """
        return round(net_cash_flow / abs(self.adquisition_cost) * 100, 2)

    def AverageCashOnCashReturn(
        self,
    ) -> float:
        """Average Cash on Cash Return of the investment during the term of the
        loan.

        Returns
        -------
        float
            Percentage that represents the Average Cash on Cash return of
            investment.
        """
        return round(statistics.mean(self.YearlyCashOnCashReturn()), 2)

    def IRR(
        self,
    ) -> float:
        """Internal Rate of Return. The IRR is the interest rate (also known as
        the discount rate) that will bring a series of cash flows (positive and
        negative) to a net present value (NPV) of zero (or to the current value
        of cash invested).

        Returns
        -------
        float
            Percentage that represent the IRR of the investment.
        """
        return round(npf.irr(self.levered_net_cash_flows) * 100, 2)

    def EquityMultiple(
        self,
    ) -> float:
        """
        """
        return round( 1 + sum(self.levered_net_cash_flows) / abs(self.adquisition_cost), 2)

    def YearlyNetCashFlowProjection(self) -> List:
        """Porjected performance of the property during the term of the loan.

        Returns
        -------
        List[Dict[str, Any]]
            Net Chasflows of the properties year by year.
        """
        yearly_net_cash_flow = [
            {"NetCashFlow": self.adquisition_cost},
        ]

        revenue: Union[int, float] = self.deal_metrics.initial_revenue
        expenses: Union[int, float] = self.deal_metrics.initial_expenses
        capital_reserve: Union[int, float] = self.deal_metrics.initial_capital_reserve
        debt_payment: float = self.loan_terms.yearly_loan_payment
        for i in range(1, self.loan_terms.term + 1, 1):
            noi: Union[int, float] = round(revenue + expenses, 2)
            net_cash_flow: Union[int, float] = round(noi + capital_reserve + debt_payment, 2)
            cash_on_cash_return = self.CashOnCashReturn(net_cash_flow)
            year = {
                "Year": i,
                "Revenue": revenue,
                "Expenses": expenses,
                "NOI": noi,
                "CapitalReserve": capital_reserve,
                "NetCashFlow": net_cash_flow,
                "CashOnCashReturn": cash_on_cash_return,
            }
            yearly_net_cash_flow.append(year)
            revenue += round(revenue * self.deal_metrics.annual_revenue_growth, 2)
            revenue = round(revenue, 2)
            expenses += round(expenses * self.deal_metrics.annual_expense_growth, 2)
            expenses = round(expenses, 2)
            capital_reserve += round(capital_reserve * self.deal_metrics.annual_capital_reserve_growth, 2)
            capital_reserve = round(capital_reserve, 2)

        after_term_noi = round(revenue + expenses, 2)

        # Adding the cashflow when the CRE is sold at the end of the term.
        sale: Dict = yearly_net_cash_flow[-1]
        sale["NetCashFlow"] = round((
            sale["NetCashFlow"] +
            self.sale_metrics.ProjectedSalePrice(after_term_noi) +
            self.loan_terms.balloon_payment
        ), 2)
        return yearly_net_cash_flow

    def LeveredNetCashFlows(self) -> List[float]:
        levered_net_cash_flows: List[float] = [
            i["NetCashFlow"] for i in self.yearly_net_cash_flow_projection
        ]
        return levered_net_cash_flows

    def YearlyCashOnCashReturn(self) -> List[float]:
        yearly_cash_on_cash_return: List[float] = [
            i["CashOnCashReturn"] for i in self.yearly_net_cash_flow_projection
            if "CashOnCashReturn" in i
        ]
        return yearly_cash_on_cash_return


if __name__ == "__main__":

    import json

    # purchase_price = 8_000_000
    purchase_price = 6_500_000

    deal = DealMetrics(
        purchase_price=purchase_price,
        going_in_cap_rate=4.84,
        closing_and_renovations=80_000,
        net_leasable_area=25_000,
        yearly_sf_operating_revenue=27.50,
        yearly_sf_operating_expenses=12.00,
        yearly_sf_capital_reserves=0.30,
        annual_revenue_growth=3.00,
        annual_expense_growth=2.50,
        annual_capital_reserve_growth=2.50,
    )

    print("DEAL DETAILS")
    print(deal)

    loan_terms = LoanTerms(
        purchase_price=purchase_price,
        loan_to_value_ratio=70,
        loan_origination_fees=1,
        interest_rate=4.5,
        amortization=30,
        term=10,
    )

    print("LOAN TERMS")
    print(loan_terms)

    sale = SaleMetrics(
        exit_cap_rate=6.25,
        cost_of_sale=2.50,
        sale_year=10,
        percentage=True,
    )

    print("SALE END OF THE TERM")
    print(sale)

    deal_projection = ReturnOfInvestmentMetrics(
        deal_metrics=deal,
        loan_terms=loan_terms,
        sale_metrics=sale,
    )

    print("DEAL RETURN PROJECTION")
    print(deal_projection)
