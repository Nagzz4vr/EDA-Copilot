from typing import Dict, Any
from signals.rules.base_rule import BaseRule,RuleOutput

class NumericDistributionPlot(BaseRule):
    name = "R5.1_numeric_distribution_plot"
    priority = 20

    def applies(self, context) -> bool:
        # Trigger if any numeric column is NOT a constant
        return any(
            col["type"] == "numeric" and "constant" not in col["flags"]
            for col in context["columns"]
        )

    def run(self, context):
        target_cols = [
            col["name"] for col in context["columns"]
            if col["type"] == "numeric" and "constant" not in col["flags"]
        ]

        
        return RuleOutput(
                rule_name=self.name,
                priority=self.priority,
                type="visualization",
                action="generate_plots",
                severity="info",
                message=f"Generating distribution plots (histogram + boxplot) for: {', '.join(target_cols)}",
                details={
                    "columns": target_cols,
                    "plot_types": ["histogram", "boxplot"]
                }
                        )
    
class SkewedNumericLogPlot(BaseRule):
    name = "R5.2_skewed_numeric_log_plot"
    priority = 15

    def applies(self, context) -> bool:
        # Trigger specifically for columns already flagged as 'skewed'
        return any(
            col["type"] == "numeric" and "skewed" in col["flags"]
            for col in context["columns"]
        )

    def run(self, context):
        target_cols = [
            col["name"] for col in context["columns"]
            if col["type"] == "numeric" and "skewed" in col["flags"]
        ]

        return RuleOutput(
                rule_name=self.name,
                priority=self.priority,
                type="visualization",
                action="generate_plots",
                severity="info",
                message=f"Highly skewed data detected. Adding log-scale histograms for: {', '.join(target_cols)} to better visualize the long tail.",
                details={
                    "columns": target_cols,
                    "plot_types": ["log_histogram"]
                }
                        )           

class CategoricalBarChart(BaseRule):
    name = "R5.3_categorical_bar_chart"
    priority = 20

    def applies(self, context) -> bool:
        return any(
            col["type"] == "categorical" and col["cardinality"] <= 20
            for col in context["columns"]
        )

    def run(self, context):
        target_cols = [
            col["name"] for col in context["columns"]
            if col["type"] == "categorical" and col["cardinality"] <= 20
        ]

        return RuleOutput(
                rule_name=self.name,
                priority=self.priority,
                type="visualization",
                action="generate_plots",
                severity="info",
                message=f"Generating full bar charts for categories: {', '.join(target_cols)}",
                details={
                    "columns": target_cols,
                    "mode": "full",
                    "plot_types": ["bar_chart"]
                }
                        )
    
class HighCardinalityBarChart(BaseRule):
    name = "R5.4_high_cardinality_bar_chart"
    priority = 15

    def applies(self, context) -> bool:
        return any(
            col["type"] == "categorical" and col["cardinality"] > 20
            for col in context["columns"]
        )

    def run(self, context):
        target_cols = [
            col["name"] for col in context["columns"]
            if col["type"] == "categorical" and col["cardinality"] > 20
        ]

        return RuleOutput(
                rule_name=self.name,
                priority=self.priority,
                type="visualization",
                action="generate_plots",
                severity="info",
                message=(
                    f"High cardinality detected in {', '.join(target_cols)}. "
                    "Plotting Top 10 categories and grouping the remainder into 'Other'."
                ),
                details={
                    "columns": target_cols,
                    "mode": "top_10_grouped",
                    "grouping_label": "Other",
                    "plot_types": ["bar_chart"]
                }
                        )
    
class CorrelationHeatmapTrigger(BaseRule):
    name = "R5.5_correlation_heatmap_trigger"
    priority = 10

    def applies(self, context) -> bool:
        # Check if we have at least 3 numeric columns
        numeric_cols = [col for col in context["columns"] if col["type"] == "numeric"]
        return len(numeric_cols) >= 3

    def run(self, context):
        numeric_cols = [col["name"] for col in context["columns"] if col["type"] == "numeric"]
        
        return RuleOutput(
                rule_name=self.name,
                priority=self.priority,
                type="visualization",
                action="generate_plots",
                severity="info",
                message=f"Dataset contains {len(numeric_cols)} numeric features. Generating a global correlation heatmap to identify multicollinearity clusters.",
                details={
                    "columns": numeric_cols,
                    "plot_types": ["correlation_heatmap"]
                }
                        )
    
class MissingDataHeatmapTrigger(BaseRule):
    name = "R5.6_missing_data_heatmap_trigger"
    priority = 10

    def applies(self, context) -> bool:
        # Count columns where missing percentage is above 5%
        messy_cols = [
            col for col in context["columns"] 
            if col["missing"]["percent"] > 5
        ]
        return len(messy_cols) >= 5

    def run(self, context):
        messy_cols = [
            col["name"] for col in context["columns"] 
            if col["missing"]["percent"] > 5
        ]

        return RuleOutput(
                rule_name=self.name,
                priority=self.priority,
                type="visualization",
                action="generate_plots",
                severity="info",
                message=(
                    f"Significant missingness found across {len(messy_cols)} columns. "
                    "Generating a nullity heatmap to check for systematic patterns of data loss."
                ),
                details={
                    "columns": messy_cols,
                    "plot_types": ["missing_data_heatmap"]
                }
                        )