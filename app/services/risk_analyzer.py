class RiskAnalyzer:
    def __init__(self, monthly_income: float, monthly_expenses: float, current_balance: float):
        """
        Initialize the analyzer with the user's basic financial metrics.
        These data points will eventually be fetched from our PostgreSQL database.
        """
        self.monthly_income = monthly_income
        self.monthly_expenses = monthly_expenses
        self.current_balance = current_balance
        self.free_cash_flow = monthly_income - monthly_expenses

    def assess_purchase_risk(self, item_name: str, item_price: float, is_credit: bool = False, credit_months: int = 1) -> dict:
        """
        Analyzes whether it is safe to purchase an item (either outright or on credit).
        """
        # Scenario 1: Outright purchase (no credit)
        if not is_credit:
            if item_price > self.current_balance:
                return {
                    "is_risky": True,
                    "reason": "critical_funds",
                    "details": f"Insufficient funds. Balance: {self.current_balance}, Price: {item_price}."
                }
            
            # If the purchase consumes the entire free cash flow for this month
            if item_price > self.free_cash_flow:
                return {
                    "is_risky": True,
                    "reason": "cashflow_warning",
                    "details": f"Purchase will consume all free cash flow this month. Remaining: {self.free_cash_flow - item_price:.2f}."
                }
            
            return {
                "is_risky": False, 
                "reason": "safe", 
                "details": "The purchase is absolutely safe for your budget."
            }

        # Scenario 2: Credit purchase (e.g., installments)
        monthly_payment = item_price / credit_months
        new_total_expenses = self.monthly_expenses + monthly_payment
        
        # Calculate DTI (Debt-to-Income Ratio).
        # Banks generally consider DTI > 50% (0.5) to be risky.
        dti_ratio = new_total_expenses / self.monthly_income

        if dti_ratio > 0.8:
            return {
                "is_risky": True,
                "reason": "dti_critical",
                "details": f"Critical risk! With this credit, your expenses will be {dti_ratio * 100:.1f}% of your income."
            }
        elif self.free_cash_flow < monthly_payment:
            return {
                "is_risky": True,
                "reason": "negative_cashflow",
                "details": f"You cannot afford the monthly payment. Free cash flow is {self.free_cash_flow:.2f}, but payment is {monthly_payment:.2f}."
            }

        return {
            "is_risky": False,
            "reason": "manageable_credit",
            "details": f"Credit is manageable. Your monthly payment will be {monthly_payment:.2f}."
        }