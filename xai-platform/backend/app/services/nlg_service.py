import openai
from typing import Dict, Any, List, Optional
from app.config import settings

class NLGService:
    """Natural Language Generation service for explaining SHAP outputs."""

    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY) if hasattr(settings, 'OPENAI_API_KEY') else None

    def generate_local_explanation(
        self,
        shap_values: List[float],
        feature_names: List[str],
        feature_values: List[Any],
        base_value: float,
        prediction: float,
        prediction_label: str = None,
        num_features: int = 5
    ) -> str:
        """
        Generate a plain-language explanation for a single prediction using SHAP values.

        Args:
            shap_values: SHAP values for each feature
            feature_names: Names of the features
            feature_values: Actual values of the features in this prediction
            base_value: Base value (average model output over training data)
            prediction: Model's prediction value
            prediction_label: Classification label (if applicable)
            num_features: Number of top features to include

        Returns:
            Natural language explanation string
        """
        # Sort features by absolute SHAP value
        indexed = list(enumerate(shap_values))
        sorted_indices = sorted(indexed, key=lambda x: abs(x[1]), reverse=True)[:num_features]

        # Build feature contribution list
        contributions = []
        for idx, shap_val in sorted_indices:
            feature_name = feature_names[idx]
            feature_value = feature_values[idx] if idx < len(feature_values) else "N/A"
            direction = "increased" if shap_val > 0 else "decreased"
            contributions.append(f"- {feature_name}: {feature_value} (impact: {shap_val:+.4f}) — this {direction} the prediction")

        contributions_text = "\n".join(contributions)

        # Build prompt
        if prediction_label:
            prompt = f"""The model predicted: {prediction_label} with a score of {prediction:.4f}.

The top contributing factors were:
{contributions_text}

Base value (average prediction): {base_value:.4f}

Generate a clear, 2-3 sentence explanation for a non-technical user. Focus on what factors were most important and whether they increased or decreased the prediction."""
        else:
            prompt = f"""The model predicted a value of {prediction:.4f}.

The top contributing factors were:
{contributions_text}

Base value (average prediction): {base_value:.4f}

Generate a clear, 2-3 sentence explanation for a non-technical user. Focus on what factors were most important and whether they increased or decreased the prediction."""

        try:
            if self.client:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an AI explainability expert that makes technical SHAP explanations understandable to everyday users."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=200
                )
                return response.choices[0].message.content.strip()
            else:
                # Fallback to template-based explanation
                return self._generate_template_explanation(prediction, base_value, contributions_text, prediction_label)

        except Exception as e:
            # Fallback to template if API fails
            return self._generate_template_explanation(prediction, base_value, contributions_text, prediction_label)

    def generate_global_explanation(
        self,
        feature_importance: List[Dict[str, Any]],
        model_name: str,
        task_type: str = "classification"
    ) -> str:
        """
        Generate a plain-language summary of global feature importance.
        """
        top_features = feature_importance[:10]

        features_text = "\n".join([
            f"- {feat['feature']}: {feat['importance']:.4f}"
            for feat in top_features
        ])

        prompt = f"""For the model '{model_name}' (task type: {task_type}), the most important features are:

{features_text}

Write a 2-3 sentence summary explaining which factors matter most for this model's predictions and why that might be important for users to understand."""

        try:
            if self.client:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an AI explainability expert that makes technical feature importance understandable."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=200
                )
                return response.choices[0].message.content.strip()
            else:
                return self._generate_global_template(model_name, task_type, top_features)

        except Exception:
            return self._generate_global_template(model_name, task_type, top_features)

    def _generate_template_explanation(
        self,
        prediction: float,
        base_value: float,
        contributions_text: str,
        prediction_label: str = None
    ) -> str:
        """Generate a template-based explanation without LLM."""
        lines = contributions_text.split("\n")
        top_contrib = lines[0] if lines else "No feature contributions available."

        if prediction_label:
            explanation = f"""The model's prediction of '{prediction_label}' (score: {prediction:.4f}) was primarily influenced by {top_contrib.split(':')[0]}.

Starting from a baseline of {base_value:.4f}, the key factors pushed the prediction up or down as shown above.

This explanation shows how each feature contributed to the final decision."""
        else:
            explanation = f"""The model's prediction of {prediction:.4f} was primarily influenced by {top_contrib.split(':')[0]}.

Starting from a baseline of {base_value:.4f}, the key factors pushed the prediction up or down as shown above.

This explanation shows how each feature contributed to the final decision."""

        return explanation.strip()

    def _generate_global_template(
        self,
        model_name: str,
        task_type: str,
        top_features: List[Dict[str, Any]]
    ) -> str:
        """Generate a template-based global explanation without LLM."""
        if not top_features:
            return f"The model '{model_name}' does not have clear feature importance information available."

        top_feature = top_features[0]['feature']
        return f"For the '{model_name}' model, '{top_feature}' is the most important feature, followed by {top_features[1]['feature'] if len(top_features) > 1 else 'others'}.

These features have the largest impact on the model's predictions across all data points. Understanding these key drivers helps ensure the model is making decisions for the right reasons."


nlg_service = NLGService()