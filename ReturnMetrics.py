"""
CRE Return Metrics
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
# from dataclasses import dataclass

import numpy as np
import numpy_financial as npf


class DealMetrics:
    """All the information that comes from the client.
    """
    def __init__(self,
        purchase_price: int,
        closing_and_renovations: int,
        net_operating_income: int,
        annual_noi_growth: float,
        percentage: bool = False,
    ) -> None:
        self.purchase_price: int = math.floor(purchase_price)
        self.closing_and_renovations: int = math.floor(closing_and_renovations)
        self.net_operating_income: int = math.floor(net_operating_income)
        if percentage:
            self.annual_noi_growth: float = round(annual_noi_growth / 100, 5)
        else:
            self.annual_noi_growth: float = round(annual_noi_growth, 5)

    def __str__(self) -> str:
        deal_metrics: Dict = {
            "PurchasePrice": self.purchase_price,
            "ClosinAndRenovations": self.closing_and_renovations,
            "NetOperatinIncome": self.net_operating_income,
            "AnnualNOIGrowth(%)": round(self.annual_noi_growth * 100, 2)
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
        percentage: bool = False,
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
        percentage: bool = False,
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
    """
    def __init__(
        self,
        deal_metrics: DealMetrics,
        loan_terms: LoanTerms,
        sale_metrics: Optional[SaleMetrics] = None,
    ) -> None:
        self.deal_metrics: DealMetrics = deal_metrics
        self.loan_terms: LoanTerms = loan_terms
        self.sale_metrics: Optional[SaleMetrics]  = sale_metrics

        # CALCULATED FIELDS
        self.adquisition_cost: float = self.AdquisitionCost()
        # self.cap_rate: float = self.CapRate()

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

    def CapRate(self) -> float:
        """Capitalization (Cap) Rate. Yield on net operating income based on
        the purchase price without taking into account debt or capital expenses
        on the deal.

        Returns
        -------
        float
        """
        cap_rate = (
            self.deal_metrics.net_operating_income
            /
            self.deal_metrics.purchase_price
        )
        cap_rate = round(cap_rate, 4)
        return cap_rate

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
        yearly_cash_on_cash_return: List[float],
    ) -> float:
        """Average Cash on Cash Return of the investment during the term of the
        loan.

        Parameters
        ----------
        yearly_cash_on_cash_return: List[float]
            List of all the cash on cash returns calculated through the term of
            the loan.

        Returns
        -------
        float
            Percentage that represents the Average Cash on Cash return of
            investment.
        """
        return round(statistics.mean(yearly_cash_on_cash_return), 2)

    def IRR(
        self,
        levered_net_cash_flows: List[float],
    ) -> float:
        """Internal Rate of Return. The IRR is the interest rate (also known as
        the discount rate) that will bring a series of cash flows (positive and
        negative) to a net present value (NPV) of zero (or to the current value
        of cash invested).

        Parameters
        ----------
        levered_net_cash_flows: List[float]
            List of all the net cash flows from the CRE investment during the
            term of the loan, starting always with the adquisition costs.

        Returns
        -------
        float
            Percentage that represent the IRR of the investment.
        """
        return round(npf.irr(levered_net_cash_flows) * 100, 2)

    def EquityMultiple(
        self,
        levered_net_cash_flows: List[float],
    ) -> float:
        """
        """
        return round(sum(levered_net_cash_flows) / abs(self.adquisition_cost), 2)

    def InvestmentReturns(self) -> Dict[str, Any]:
        """Return measurements. Creates a digestable information table
        regarding the different return metrics of the deal.

        Returns
        -------
        Dict[str, Any]
        """
        deal: Dict = {
            "PurchasePrice": self.deal_metrics.purchase_price,
            "ClosingCosts": self.deal_metrics.closing_and_renovations,
            "LoanProceeds": self.loan_terms.loan_ammount,
            "LoanOriginationFees(%)": round(self.loan_terms.loan_origination_fees * 100, 2),
            "AdquisitionCost": self.adquisition_cost,
            "BalloonPayment": self.loan_terms.balloon_payment,
            "AverageCashOnCashReturn(%)": 0,
            "IRR(%)": 0,
            "EquityMultiple": 0,
            # where we are going to store the values for each year of operation
            # during the term
            "YearlyCashOnCashReturn": [],
        }

        net_operating_income: Union[int, float] = self.deal_metrics.net_operating_income
        yearly_loan_payment: float = self.loan_terms.yearly_loan_payment

        cash_on_cash_return: List[float] = []

        if self.sale_metrics:
            if self.loan_terms.term != self.sale_metrics.sale_year:
                raise ValueError("The year of selling is different to the term of the loan.")

        for i in range(1, self.loan_terms.term + 1, 1):
            net_cash_flow = net_operating_income + yearly_loan_payment
            year: Dict = {
                "Year": i,
                "CashFlow": net_operating_income,
                "LoanPayment": yearly_loan_payment,
                "NetCashFlow": round(net_cash_flow, 2),
                "CashOnCashReturn(%)": self.CashOnCashReturn(net_cash_flow),
            }
            deal["YearlyCashOnCashReturn"].append(year)
            cash_on_cash_return.append(year["CashOnCashReturn(%)"])
            net_operating_income += round(net_operating_income * self.deal_metrics.annual_noi_growth, 2)
            # In the last iteration, the NOI variable keeps the value of the
            # year after the term. This is the value that we need to calculate
            # the selling price of the property at the end of the term.
            net_operating_income = round(net_operating_income, 2)

        # Average CoC
        deal["AverageCashOnCashReturn(%)"] = self.AverageCashOnCashReturn(cash_on_cash_return)

        # this is what happends if there is a sale at the end of the term.
        if self.sale_metrics:
            sale: Dict = deal["YearlyCashOnCashReturn"][-1]

            # the idea of this code is to add here the changes in the last
            # cashflow when we sell the property, which would be,
            # NetCashFlow of that year + calculated sale price of the property
            # + the BalloonPayment of the deal at the end of the term.
            sale["NetCashFlow"] = round((
                sale["NetCashFlow"] +
                self.sale_metrics.ProjectedSalePrice(net_operating_income) +
                deal["BalloonPayment"]
            ), 2)

        # IRR
        # This is a hack to make the creation of the list faster. We create the
        # list of all the yearlys NetCashFlow and then, we insert at the
        # begining of the list the AdquisitionCost of the deal. The order
        # matters.
        levered_net_cash_flows = [
                i["NetCashFlow"] for i in deal["YearlyCashOnCashReturn"]
        ]
        levered_net_cash_flows.insert(0, deal["AdquisitionCost"])
        deal["IRR(%)"] = self.IRR(levered_net_cash_flows)

        # EquityMultiple
        levered_net_cash_flows = [
                i["NetCashFlow"] for i in deal["YearlyCashOnCashReturn"]
        ]
        deal["EquityMultiple"] = self.EquityMultiple(levered_net_cash_flows)
        return deal


if __name__ == "__main__":

    import json

    new_deal = DealMetrics(
        purchase_price = 10_000_000,
        closing_and_renovations = 100_000,
        net_operating_income = 600_000,
        annual_noi_growth = 3.0,
        percentage = True,
    )

    print("NEW DEAL")
    print(new_deal)

    loan_terms = LoanTerms(
        purchase_price = 10_000_000,
        loan_to_value_ratio = 70,
        loan_origination_fees = 1,
        interest_rate = 4.5,
        amortization = 30,
        term = 10,
        percentage = True,
    )

    print("LOAN TERMS")
    print(loan_terms)

    new_sale = SaleMetrics(
        exit_cap_rate=6.75,
        cost_of_sale=2.50,
        sale_year=10,
        percentage=True,
    )

    print("NEW SALE")
    print(new_sale)


    deal_metrics = ReturnOfInvestmentMetrics(
        deal_metrics=new_deal,
        loan_terms=loan_terms,
        # sale_metrics=new_sale,
    )


    string = json.dumps(deal_metrics.InvestmentReturns(), indent=4)
    print(string)

    # new_deal = DealMetrics(
    #     purchase_price = 10_000_000,
    #     closing_and_renovations = 100_000,
    #     net_operating_income = 600_000,
    #     annual_noi_growth = 3.0,
    #     percentage = True,
    # )

    # print("NEW DEAL")
    # print(new_deal)

    # loan_terms = LoanTerms(
    #     loan_to_value_ratio = 0,
    #     loan_ammount = 0,
    #     loan_origination_fees = 0,
    #     interest_rate = 0,
    #     amortization = 0,
    #     term = 10,
    #     percentage = True,
    # )

    # print("LOAN TERMS")
    # print(loan_terms)

    # deal_metrics = ReturnOfInvestmentMetrics(
    #     deal_metrics=new_deal,
    #     loan_terms=loan_terms,
    # )

    # string = json.dumps(deal_metrics.CashOnCashReturn(), indent=4)

    # print(string)
