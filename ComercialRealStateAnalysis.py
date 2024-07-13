"""
CRE Valuation and Return Of Investment Metrics
"""
from typing import (
    Union,
    Iterable,
    List,
    Dict,
)
import math
import statistics

import numpy as np
import numpy_financial as npf


class TaxAssumptions:
    def __init__(
        self,
        property_type: str,
        lan_vs_building_value: Union[int, float],
        depreciation_timeline: Union[int, float],
        income_tax_rate: Union[int, float],
        capital_gains_tax_rate: Union[int, float],
        depreciation_recapture_tax_rate: Union[int, float],
        percentage: bool = True,
    ) -> None:
        """All the information for the return calculations taking taxes into
        account.

        Parameters
        ----------
        property_type: str
            Type of property bought or about to be bought.
        lan_vs_building_value: int | float
            How much percentage of the property value is allocated to land.
        depreciation_timeline: int | float
            In how many years the building (without the land) will depreciate
            completely to 0.
        income_tax_rate: int | float
            Tax rate over the income of the property
        capital_gains_tax_rate: int | float
            Tax rate over the capital gains at the selling of the property.
        depreciation_recapture_tax_rate: int | float
            Tax rate over the total ammount of the devaluation of the building
            that will be recovered at the time of the sale.
        """
        self.property_type: str = property_type
        self.depreciation_timeline: float = round(depreciation_timeline, 2)

        if percentage:
            self.lan_vs_building_value: float = round(lan_vs_building_value / 100, 4)
            self.income_tax_rate:float = round(income_tax_rate / 100, 4)
            self.capital_gains_tax_rate:float = round(capital_gains_tax_rate / 100, 4)
            self.depreciation_recapture_tax_rate:float = round(depreciation_recapture_tax_rate / 100, 4)
        else:
            self.lan_vs_building_value: float = round(lan_vs_building_value, 4)
            self.income_tax_rate:float = round(income_tax_rate, 4)
            self.capital_gains_tax_rate:float = round(capital_gains_tax_rate, 4)
            self.depreciation_recapture_tax_rate:float = round(depreciation_recapture_tax_rate, 4)

    def __str__(self) -> str:
        string: Dict = {
            "PropertyType": self.property_type.upper(),
            "LandValueProportion": self.lan_vs_building_value,
            "DepreciationTimeline": self.depreciation_timeline,
            "IncomeTaxRate": round(self.income_tax_rate * 100, 2),
            "CapitalGainsTaxRate": round(self.capital_gains_tax_rate * 100, 2),
            "DepreciationRecaptureTaxRate": round(self.depreciation_recapture_tax_rate * 100, 2),
        }
        return json.dumps(string, indent=4)


class CREInformation:
    def __init__(
        self,
        net_leasable_area: Union[int, float],
        initial_yearly_sf_operating_revenue: Union[int, float],
        initial_yearly_sf_operating_expenses: Union[int, float],
        initial_yearly_sf_capital_reserves: Union[int, float],
    ) -> None:
        self.net_leasable_area: Union[int, float] = round(net_leasable_area, 2)
        self.initial_yearly_sf_operating_revenue: Union[int, float] = round(
            initial_yearly_sf_operating_revenue,
            2,
        )
        self.initial_yearly_sf_operating_expenses: Union[int, float] = round(
            initial_yearly_sf_operating_expenses,
            2,
        )
        self.initial_yearly_sf_capital_reserves: Union[int, float] = round(
            initial_yearly_sf_capital_reserves,
            2,
        )

    def __str__(self) -> str:
        string: Dict = {
            "NetLeasableArea": self.net_leasable_area,
            "Year1SFOperatingRevenue": self.initial_yearly_sf_operating_revenue,
            "Year1SFOperatingExpenses": self.initial_yearly_sf_operating_expenses,
            "Year1SFCapitalReserves": self.initial_yearly_sf_capital_reserves,
        }
        return json.dumps(string, indent=4)

    def CREYearlyRevenue(self) -> float:
        """Yearly Revenue Per Square Feet multiplied by the Net Leasable Area.
        """
        return round(
            self.net_leasable_area * self.initial_yearly_sf_operating_revenue,
            2,
        )

    def CREYearlyExpenses(self) -> float:
        """Yearly Expenses Per Square Feet multiplied by the Net Leasable Area.
        """
        return round(
            self.net_leasable_area * self.initial_yearly_sf_operating_expenses,
            2,
        )

    def CREYearlyCapitalReserves(self) -> float:
        """Yearly Capital Reserves Per Square Feet multiplied by the Net
        Leasable Area.
        """
        return round(
            self.net_leasable_area * self.initial_yearly_sf_capital_reserves,
            2,
        )


class DealMetrics:
    """All the information that comes from the client.
    """
    def __init__(self,
        purchase_price: Union[int, float],
        going_in_cap_rate: float,
        closing_and_renovations: int,
        initial_yearly_revenue: Union[int, float],
        initial_yearly_operating_expenses: Union[int, float],
        initial_yearly_capital_reserves: Union[int, float],
        annual_revenue_growth: float,
        annual_expense_growth: float,
        annual_capital_reserve_growth: float,
        percentage: bool = True,
    ) -> None:
        self.purchase_price: int = math.floor(purchase_price)
        self.closing_and_renovations: int = math.floor(closing_and_renovations)

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

        self.initial_revenue: Union[int, float] = round(initial_yearly_revenue, 2)
        self.initial_expenses: Union[int, float] = - round(initial_yearly_operating_expenses, 2)
        self.initial_capital_reserve: Union[int, float] = - round(initial_yearly_capital_reserves, 2)

    def __str__(self) -> str:
        """JSON string like that represents the initial metricss of the deal.
        """
        deal_metrics: Dict = {
            "PurchasePrice": self.purchase_price,
            "GoingInCapRate(%)": round(self.going_in_cap_rate * 100, 2),
            "ClosingAndRenovations": self.closing_and_renovations,
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
        purchase_price: Union[int, float],
        loan_to_value_ratio: float,
        loan_origination_fees: float,
        interest_rate: float,
        amortization: int,
        term: int,
        interest_only_period: int = 0,
        future_value: int = 0,
        percentage: bool = True,
    ) -> None:

        self.amortization: int = math.floor(amortization)
        self.term: int = math.floor(term)
        self.interest_only_period: int = math.floor(interest_only_period)

        if percentage:
            self.loan_to_value_ratio: float = round(loan_to_value_ratio / 100, 4)
            self.loan_origination_fees: float = round(loan_origination_fees / 100, 4)
            self.interest_rate: float = round(interest_rate / 100, 4)
        else:
            self.loan_to_value_ratio: float = round(loan_to_value_ratio, 4)
            self.loan_origination_fees: float = round(loan_origination_fees, 4)
            self.interest_rate: float = round(interest_rate, 4)

        # CALCULATED FIELDS
        self.loan_ammount: float = round(math.floor(purchase_price) * self.loan_to_value_ratio, 2)
        self.yearly_loan_payment: float = self.YearlyLoanPayment(future_value)
        self.yearly_loan_principal_payments: Iterable[float] = self.YearlyLoanPrincipalPayments(future_value)
        self.yearly_loan_interest_payments: Iterable[float] = self.YearlyLoanInterestPayments(future_value)
        self.balloon_payment: float = self.BalloonPayment(future_value)


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

    def InterestOnlyPayment(self) -> float:
        payment = self.loan_ammount * self.interest_rate
        return round(- payment, 2)

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

    def MonthlyLoanInterestPayments(
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

    def YearlyLoanInterestPayments(
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
        interest_payments = self.MonthlyLoanInterestPayments(future_value)

        yearly_interest_payments: List[float] = []
        for i in range(self.term):
            year = round(sum(interest_payments[i*12 : (i+1)*12]), 2)
            yearly_interest_payments.append(year)
        for i in range(self.interest_only_period):
            interest_only_payment = self.InterestOnlyPayment()
            yearly_interest_payments.insert(i, interest_only_payment)
        return yearly_interest_payments[0 : self.term]

    def MonthlyLoanPrincipalPayments(
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

        principal_payments = npf.ppmt(
            rate=self.interest_rate / 12,
            nper= self.amortization * 12,
            per=np.arange(self.amortization * 12) + 1,
            pv=self.loan_ammount,
            fv=future_value,
        )

        principal_payments = np.round(principal_payments, 2)
        return principal_payments

    def YearlyLoanPrincipalPayments(
        self,
        future_value: int = 0,
    ) -> List[float]:
        """Principal paid yearly to the bank for the term of the loan.

        Parameters
        ----------
        future_value: int | float
            Money owed after the last payment. By default is 0.

        Returns
        -------
        List[float]
        """
        principal_payments = self.MonthlyLoanPrincipalPayments(future_value)

        yearly_principal_payments: List[float] = []
        for i in range(self.term):
            year = round(sum(principal_payments[i*12 : (i+1)*12]), 2)
            yearly_principal_payments.append(year)
        for i in range(self.interest_only_period):
            yearly_principal_payments.insert(i, 0.0)
        return yearly_principal_payments[0 : self.term]


    def BalloonPayment(
        self,
        future_value: int = 0,
    ) -> float:
        """Balloon payment of the loan at the end of the term.

        Returns
        -------
        float
        """
        paid_to_the_bank: float = sum(self.yearly_loan_principal_payments)
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
        tax_assumptions: TaxAssumptions,
    ) -> None:
        self.deal_metrics: DealMetrics = deal_metrics
        self.loan_terms: LoanTerms = loan_terms
        self.sale_metrics: SaleMetrics  = sale_metrics
        self.tax_assumptions: TaxAssumptions = tax_assumptions

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
            "IRR(%)": round(self.irr * 100, 2),
            "EquityMultiple": self.equity_multiple,
            "AverageCashOnCashReturn(%)": round(self.average_cash_on_cash_return * 100, 2),
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
        return round(net_cash_flow / abs(self.adquisition_cost), 4)

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
        return round(statistics.mean(self.YearlyCashOnCashReturn()), 4)

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
        return round(npf.irr(self.levered_net_cash_flows), 4)

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

        # getting the building value
        purchase_price: int = self.deal_metrics.purchase_price
        val_percentage_land: float = self.tax_assumptions.lan_vs_building_value
        building_value: float = round(purchase_price*(1-val_percentage_land),2)

        # depreciation of the building
        depre_time: float = self.tax_assumptions.depreciation_timeline
        depreciation: float = round(-building_value/depre_time, 2)

        # Income tax rate
        itr: float = self.tax_assumptions.income_tax_rate

        for i in range(0, self.loan_terms.term, 1):
            # net operating income
            noi: float = round(revenue + expenses, 2)

            # debt service
            current_pp: float = self.loan_terms.yearly_loan_principal_payments[i]
            current_ip: float = self.loan_terms.yearly_loan_interest_payments[i]
            debt_payment: float = current_pp + current_ip

            # cashflow after debt service
            cfads: float = round(noi + capital_reserve + debt_payment, 2)

            # depreciation expenses
            depreciation_expense = depreciation if i < depre_time else 0.0

            # income taxes
            it = round(-(noi + current_ip + depreciation_expense) * itr, 2)
            implied_tax_rate: float = round(abs(it/cfads), 4)

            # Net Cash Flow
            ncf: float = round(cfads + it, 2)

            # cash on cash return
            cocr: float = self.CashOnCashReturn(ncf)

            year = {
                "Year": i + 1,
                "Revenue": revenue,
                "Expenses": expenses,
                "NOI": noi,
                "CapitalReserve": capital_reserve,
                "PrincipalPayment": current_pp,
                "InterestPayment": current_ip,
                "CashFlowAfterDebtService": cfads,
                "DepreciationExpense": depreciation_expense,
                "IncomeTaxes": it,
                "NetCashFlow": ncf,
                "ImpliedTaxRate": implied_tax_rate,
                "CashOnCashReturn": cocr,
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

        # sale price with the projected NOI
        sale_price = self.sale_metrics.ProjectedSalePrice(after_term_noi)
        print(sale_price)
        # Capital gains taxes
        closing_and_renovations = self.deal_metrics.closing_and_renovations
        capital_gains: float = round(sale_price - purchase_price - closing_and_renovations, 2)
        cgt: float = - round(capital_gains * self.tax_assumptions.capital_gains_tax_rate, 2)
        print(cgt)
        # Depreciation Recapture Tax
        drtr = self.tax_assumptions.depreciation_recapture_tax_rate
        drt = round(depreciation * self.sale_metrics.sale_year * drtr, 2)
        print(drt)

        sale: Dict = yearly_net_cash_flow[-1]
        sale["NetCashFlow"] = round((
            sale["NetCashFlow"] +
            self.sale_metrics.ProjectedSalePrice(after_term_noi) +
            drtr +
            cgt +
            self.loan_terms.balloon_payment
        ), 2)
        sale["SalePrice"] = sale_price
        sale["DepreciationRecaptureTax"] = drt
        sale["CapitalGainsTax"] = cgt
        return yearly_net_cash_flow

    def LeveredNetCashFlows(self) -> List[float]:
        levered_net_cash_flows: List[float] = [
            i["NetCashFlow"] for i in self.yearly_net_cash_flow_projection
        ]
        return levered_net_cash_flows

    def YearlyCashOnCashReturn(self) -> List[float]:
        yearly_cash_on_cash_return: List[float] = [
            year["CashOnCashReturn"] for year in self.yearly_net_cash_flow_projection
            if "CashOnCashReturn" in year
        ]
        return yearly_cash_on_cash_return

    def ReturnMetricsOfInvestment(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {
            "IRR": self.irr,
            "EquityMultiple": self.equity_multiple,
            "AverageCashOnCashReturn": self.average_cash_on_cash_return,
        }
        return metrics

class TargetInvestmentMetrics:
    def __init__(
        self,
        irr: float,
        acocr: float,
        eqm: float,
        percentage: bool = True,
    ) -> None:
        if percentage:
            self.irr: float = round(irr / 100, 4)
            self.acocr: float = round(acocr / 100, 4)
        else:
            self.irr: float = round(irr, 4)
            self.acocr: float = round(acocr, 4)

        self.eqm = round(eqm, 4)

    def __str__(self) -> str:
        string: Dict = {
            "IRR(%)": round(self.irr * 100, 2),
            "EquityMultiple": self.eqm,
            "AverageCashOnCashReturn(%)": round(self.acocr * 100,2),
        }
        return json.dumps(string, indent=4)

    def TargetMetricsOfInvestment(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {
            "IRR": self.irr,
            "EquityMultiple": self.eqm,
            "AverageCashOnCashReturn": self.acocr,
        }
        return metrics

    def GradientDescentPurchasePriceMaximizer(
        self,
        initial_pp: Union[int, float],
        initial_dm: DealMetrics,
        initial_lm: LoanTerms,
        sm: SaleMetrics,
        taxes: TaxAssumptions,
        max_iterations: int = 1_000_000,
        learning_rate = 100_000,
        tolerance: float = 0.005,
    ) -> Dict[str, float]:
        """Returns that maximum purchase price of a property, given the target
        metrics.
        """
        purchase_price: Union[int, float] = initial_pp
        Expected_ROI: Dict[str, float] = self.TargetMetricsOfInvestment()
        iterations: int = 0

        while max_iterations > iterations:

            deal_metrics: DealMetrics = DealMetrics(
                purchase_price=purchase_price,
                going_in_cap_rate=initial_dm.going_in_cap_rate,
                closing_and_renovations=initial_dm.closing_and_renovations,
                initial_yearly_revenue=initial_dm.initial_revenue,
                initial_yearly_operating_expenses=-initial_dm.initial_expenses,
                initial_yearly_capital_reserves=-initial_dm.initial_capital_reserve,
                annual_revenue_growth=initial_dm.annual_revenue_growth,
                annual_expense_growth=initial_dm.annual_expense_growth,
                annual_capital_reserve_growth=initial_dm.annual_capital_reserve_growth,
                percentage=False,
            )

            loan_terms: LoanTerms = LoanTerms(
                purchase_price=purchase_price,
                loan_to_value_ratio=initial_lm.loan_to_value_ratio,
                loan_origination_fees=initial_lm.loan_origination_fees,
                interest_rate=initial_lm.interest_rate,
                amortization=initial_lm.amortization,
                term=initial_lm.term,
                interest_only_period=initial_lm.interest_only_period,
                percentage=False,
            )

            sale_metrics: SaleMetrics = SaleMetrics(
                exit_cap_rate=sm.exit_cap_rate,
                cost_of_sale=sm.cost_of_sale,
                sale_year=sm.sale_year,
                percentage=False,
            )

            ROI: Dict[str, float] = ReturnOfInvestmentMetrics(
                deal_metrics=deal_metrics,
                loan_terms=loan_terms,
                sale_metrics=sale_metrics,
                tax_assumptions=taxes,
            ).ReturnMetricsOfInvestment()

            ROI_DIFF = {
                key: Expected_ROI[key] - ROI[key] for key in Expected_ROI.keys()
            }

            if all(abs(diff) <= tolerance for diff in ROI_DIFF.values()):
                ROI["PurchasePrice"] = purchase_price
                return ROI

            avg_difference: float = round(sum(ROI_DIFF.values()) / len(ROI_DIFF), 4)

            purchase_price -= learning_rate * avg_difference

            if purchase_price <= 0:
                raise ValueError("The Purchase Price is 0 or bellow.")

            iterations += 1
        else:
            raise ValueError(f"Searched value not found. Last Iteration ended in {purchase_price}")


def GradientDescentPurchasePriceMaximizer(
    initial_pp: int,
    taxes: TaxAssumptions,
    initial_dm: DealMetrics,
    initial_lm: LoanTerms,
    sm: SaleMetrics,
    tim: TargetInvestmentMetrics,
    max_iterations: int = 100_000,
    learning_rate = 10_000,
    tolerance: float = 0.0001,
) -> Dict[str, float]:
    """
    """

    purchase_price = initial_pp
    Expected_ROI: Dict[str, float] = tim.TargetMetricsOfInvestment()
    iterations = 0

    while max_iterations > iterations:

        deal_metrics: DealMetrics = DealMetrics(
            purchase_price=purchase_price,
            going_in_cap_rate=initial_dm.going_in_cap_rate,
            closing_and_renovations=initial_dm.closing_and_renovations,
            initial_yearly_revenue=initial_dm.initial_revenue,
            initial_yearly_operating_expenses=-initial_dm.initial_expenses,
            initial_yearly_capital_reserves=-initial_dm.initial_capital_reserve,
            annual_revenue_growth=initial_dm.annual_revenue_growth,
            annual_expense_growth=initial_dm.annual_expense_growth,
            annual_capital_reserve_growth=initial_dm.annual_capital_reserve_growth,
            percentage=False,
        )

        loan_terms: LoanTerms = LoanTerms(
            purchase_price=purchase_price,
            loan_to_value_ratio=initial_lm.loan_to_value_ratio,
            loan_origination_fees=initial_lm.loan_origination_fees,
            interest_rate=initial_lm.interest_rate,
            amortization=initial_lm.amortization,
            term=initial_lm.term,
            percentage=False,
        )

        sale_metrics: SaleMetrics = SaleMetrics(
            exit_cap_rate=sm.exit_cap_rate,
            cost_of_sale=sm.cost_of_sale,
            sale_year=sm.sale_year,
            percentage=False,
        )

        ROI: Dict[str, float] = ReturnOfInvestmentMetrics(
            deal_metrics=deal_metrics,
            tax_assumptions=taxes,
            loan_terms=loan_terms,
            sale_metrics=sale_metrics,
        ).ReturnMetricsOfInvestment()

        ROI_DIFF = {
            key: Expected_ROI[key] - ROI[key] for key in Expected_ROI.keys()
        }

        if all(abs(diff) <= tolerance for diff in ROI_DIFF.values()):
            ROI["PurchasePrice"] = purchase_price
            return ROI

        avg_difference: float = round(sum(ROI_DIFF.values()) / len(ROI_DIFF), 4)

        purchase_price -= learning_rate * avg_difference
        purchase_price = math.floor(purchase_price)

        if purchase_price <= 0:
            raise ValueError("The Purchase Price is 0 or bellow.")

        iterations += 1
    else:
        raise ValueError(f"Searched value not found. Last Iteration ended in {purchase_price}")


if __name__ == "__main__":

    import json

    taxes: TaxAssumptions = TaxAssumptions(
        property_type="Residential",
        lan_vs_building_value=30,
        depreciation_timeline=27.5,
        income_tax_rate=25,
        capital_gains_tax_rate=15,
        depreciation_recapture_tax_rate=25,
    )

    print("TAX DETAILS")
    print(taxes)

    purchase_price = 10_000_000

    cre: CREInformation = CREInformation(
        net_leasable_area=25_000,
        initial_yearly_sf_operating_revenue=27.50,
        initial_yearly_sf_operating_expenses=12.00,
        initial_yearly_sf_capital_reserves=0.30,
    )

    print("CRE DETAILS")
    print(cre)

    deal: DealMetrics = DealMetrics(
        purchase_price=purchase_price,
        going_in_cap_rate=6.00,
        closing_and_renovations=225_000,
        initial_yearly_revenue=600_000,
        initial_yearly_operating_expenses=0,
        initial_yearly_capital_reserves=0,
        annual_revenue_growth=3.50,
        annual_expense_growth=0,
        annual_capital_reserve_growth=0,
    )

    print("DEAL DETAILS")
    print(deal)

    loan_terms = LoanTerms(
        purchase_price=purchase_price,
        loan_to_value_ratio=65,
        loan_origination_fees=1,
        interest_rate=4.2,
        interest_only_period=2,
        amortization=30,
        term=5,
    )

    print("LOAN TERMS")
    print(loan_terms)

    sale = SaleMetrics(
        exit_cap_rate=6.50,
        cost_of_sale=2.50,
        sale_year=5,
        percentage=True,
    )

    print("SALE END OF THE TERM")
    print(sale)

    deal_projection = ReturnOfInvestmentMetrics(
        deal_metrics=deal,
        tax_assumptions=taxes,
        loan_terms=loan_terms,
        sale_metrics=sale,
    )

    print("DEAL RETURN PROJECTION")
    print(deal_projection)

    # print("DEAL ROI")
    # print(json.dumps(deal_projection.ReturnMetricsOfInvestment(), indent=4))

    # target_investment_metrics = TargetInvestmentMetrics(
    #     irr=14,
    #     eqm=3,
    #     acocr=8,
    # )

    # print("TARGET ROI")
    # print(json.dumps(target_investment_metrics.TargetMetricsOfInvestment(), indent=4))

    # max_pp: Dict = target_investment_metrics.GradientDescentPurchasePriceMaximizer(
    #     initial_pp=purchase_price,
    #     initial_dm=deal,
    #     initial_lm=loan_terms,
    #     sm=sale,
    #     learning_rate=100_000,
    #     tolerance=0.005,
    #     max_iterations=1_000,
    # )
    # print("MAXIMAZED PURCHASE PRICE FOR THE TARGET ROI")
    # print(json.dumps(max_pp, indent=4))

    # max_pp = GradientDescentPurchasePriceMaximizer(
    #     initial_pp=purchase_price,
    #     initial_dm=deal,
    #     initial_lm=loan_terms,
    #     sm=sale,
    #     tim=target_investment_metrics,
    #     learning_rate=100_000,
    #     tolerance=0.005,
    #     max_iterations=1_000,
    # )
    # print("MAXIMAZED PURCHASE PRICE FOR THE TARGET ROI")
    # print(json.dumps(max_pp, indent=4))

