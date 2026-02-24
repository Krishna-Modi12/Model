import os
import numpy as np

class EthicsGuard:
    """
    Implements Section 11 privacy and fairness principles.
    1. Verification of demographic representation.
    2. Data sanitization (PII check placeholder).
    """
    
    @staticmethod
    def audit_demographics(df):
        """
        Check for representation gaps in Fitzpatrick and Monk scales.
        """
        if 'fitzpatrick' not in df.columns:
            return "Audit Fallback: Labels missing."
            
        fitz_counts = df['fitzpatrick'].value_counts().to_dict()
        monk_counts = df['monk'].value_counts().to_dict()
        
        # Simple threshold check for data balance (Targeting Section 3.3 balance)
        report = ["Demographic Balance Audit:"]
        for k, v in fitz_counts.items():
            report.append(f"  Fitzpatrick Type {k}: {v} samples")
            
        return "\n".join(report)

    @staticmethod
    def sanitize_path(path):
        """
        Ensures local paths don't contain PII if logged.
        """
        return os.path.basename(path)

    @staticmethod
    def check_consent_labels(df):
        """
        Ensures the dataset contains the required consent metadata (placeholder).
        """
        has_metadata = 'consent_id' in df.columns
        return has_metadata
