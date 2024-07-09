"""
CRE Investing

Just learning stuff and giving ideas to my head.
"""
from typing import Union, Literal, Iterable, Dict, Any
from decimal import Decimal
from dataclasses import dataclass

# DEAL METRICS
purchase_price: Union[int, Decimal] = 10_000_000
appraised_value: Union[int, Decimal] = 10_000_000
closing_and_renovations: Union[int, Decimal] = 500_000

# LOAN TERMS
interest_rate: float = 0.0450 # percentage
net_operating_income: Union[int, float] = 500_000
payments_per_year: int = 12
"`amortization`: Amount of time on which the loan will be spreaded out"
amortization: int = 25 # years
"`term`: Time after which the loan will be fully paid"
term: int = 10 # years

# LOAN CONSTRAINS
# QUESTION: Are the loan constrains a politic company wide, or are they studied
# case by case?.
# ANSWER: They depend on the rules of the different lenders.
LTV: float = 0.65 # percentage
LTC: float = 0.70 # percentage
"`DSCR:` Annual Net Operating Income / Annual Debt Service"
DSCR: float = 1.25 # debt service coverage ratio
"`DebtYield: ` Annual Net Operating Income / Total Loan Ammount"
DebtYield: float = 0.076 # percentage


def payment(
    present_value: int,
    future_value: int = 0,
    interest_rate: Union[int,float] = interest_rate/payments_per_year,
    n_per_y: int = amortization*payments_per_year,
    payment_type: Literal["begin","end"] = "end",
) -> Union[Iterable[float], float]:
    """Present value of an investment (Whats worth the total ammount of
    payments now)

    Parameters
    ----------
    present_value: int | float
        Ammount borrowed.
    future_value: int | float
        Money owed after the last payment. By default is 0.
    interest_rate: float
        Interest rate per period (monthly).
    n_per_y: int
        Total number of payment periods in the loan. Could be, number of
        payments per year, and we multiply that for the total ammount of years
        of the loan.
    payment_type: 1 or 0
        When the payment is due. If af the end of the period (0) or at the
        beginning of the period (1). By default is 0.

    Returns
    -------
    NDArray[float64] | float64
    """
    import numpy_financial as npf

    if not payment_type in ["begin", "end"]:
        raise ValueError("Incorrect Value entered for the payment_type parameter")

    calculated_payment = npf.pmt(
        pv=present_value,
        fv=future_value,
        rate=interest_rate,
        nper=n_per_y,
        when=payment_type,
    )
    return calculated_payment


def MLA_LTV(
    appraised_value:Union[int, float]=appraised_value,
    LTV:float=LTV,
) -> Union[int, float]:
    """Maximum loan to value. This tells us how much value of the appraised
    property can be lended to the client.

    Parameters
    ----------
    appraised_value: int | float
        Value on which the property is appraised.
    LTV: int | float
        maximum loan to value ratio that the bank will give

    Returns
    -------
    int | float
    """
    MLA_LTV = appraised_value * LTV
    return MLA_LTV


def MLA_LTC(
    purchase_price:Union[int, float]=purchase_price,
    closing_and_renovations:Union[int, float]=closing_and_renovations,
    LTC:float=LTC,
) -> Union[int, float]:
    """Maximum loan to cost. This tells us how much of the total value of the
    deal can be lended to the client.

    Parameters
    ----------
    purchase_price: int | float
        Value on which the property is being bougth.
    closing_and_renovations: int | float
        Value of closing the deal and renovate the property if nescessary
    LTC: int | float
        maximum loan to cost ratio that the bank will give

    Returns
    -------
    int | float
    """
    MLA_LTC = LTC * (purchase_price + closing_and_renovations)
    return MLA_LTC


def MLA_DSCR(
    interest_rate: Union[int,float] = interest_rate/payments_per_year,
    n_per_y: int = amortization*payments_per_year,
    payment: Union[int, float] = net_operating_income/DSCR/payments_per_year,
    future_value: int = 0,
    payment_type: Literal["begin","end"] = "end",
) -> Union[Iterable[float], float]:
    """Maximum loan ammout given the DSCR constrain. For this is the present
    value of an investment (Whats worth the total ammount of payments now).

    Parameters
    ----------
    interest_rate: float
        Interest rate per period (monthly).
    n_per_y: int
        Total number of payment periods in the loan. Could be, number of
        payments per year, and we multiply that for the total ammount of years
        of the loan.
    payment: int | float
        Monthly loan payment without changes. This payment should only include
        principal and interest, no other fees or taxes. This could be the
        net operating income divided by the maximum DSCR in the loan
        constrains, and divided again by the number of payments per year.
    future_value: int | float
        Money owed after the last payment. By default is 0.
    payment_type: 1 or 0
        When the payment is due. If af the end of the period (0) or at the
        beginning of the period (1). By default is 0.

    Returns
    -------
    NDArray[float64] | float64
    """
    import numpy_financial as npf

    if not payment_type in ["begin", "end"]:
        raise ValueError("Incorrect Value entered for the payment_type parameter")

    calculated_future_value = npf.pv(
        rate=interest_rate,
        nper=n_per_y,
        pmt=-payment,
        fv=future_value,
        when=payment_type,
    )
    return calculated_future_value


def MLA_DebtYield(
    net_operating_income: Union[int, float] = net_operating_income,
) -> Union[int, float]:
    return net_operating_income/DebtYield


def main():
    return


if __name__ == "__main__":
    main()
