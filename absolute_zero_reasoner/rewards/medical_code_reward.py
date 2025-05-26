"""
Enhanced Medical Code Reward Functions
Specialized rewards for medical visualization and clinical scoring code generation
with advanced clinical logic, safety considerations, and coding best practices.
"""

import re
import ast
import json
from typing import Dict, List, Tuple, Any, Set
import numpy as np
from collections import defaultdict


def get_enhanced_medical_visualization_reward(code_snippet: str) -> float:
    """
    Enhanced medical visualization reward with more sophisticated checks.
    """
    try:
        score = 0.0
        total_checks = 15
        
        # Previous visualization checks (condensed)
        basic_viz_score = get_medical_visualization_reward(code_snippet)
        score += basic_viz_score * 0.4  # 40% from basic checks
        
        # Advanced medical visualization patterns
        advanced_patterns = [
            r'subplots.*nrows.*ncols',  # Multi-panel layouts
            r'colorbar.*label',  # Proper colorbar labeling
            r'annotation.*medical',  # Medical annotations
            r'interactive.*plot',  # Interactive visualizations
            r'plotly.*dash|dash.*plotly',  # Dashboard creation
        ]
        
        advanced_score = sum(1 for pattern in advanced_patterns 
                           if re.search(pattern, code_snippet, re.IGNORECASE))
        score += (advanced_score / len(advanced_patterns)) * 0.2
        
        # Medical-specific plot types
        medical_plot_types = [
            r'kaplan.*meier|survival.*curve',  # Survival analysis
            r'roc.*curve|auc.*plot',  # ROC curves
            r'forest.*plot',  # Meta-analysis plots
            r'bland.*altman',  # Agreement plots
            r'waterfall.*plot',  # Treatment response
        ]
        
        plot_type_score = sum(1 for pattern in medical_plot_types 
                            if re.search(pattern, code_snippet, re.IGNORECASE))
        score += (plot_type_score / len(medical_plot_types)) * 0.2
        
        # Data quality visualization
        quality_viz_patterns = [
            r'missing.*data.*plot',
            r'outlier.*detection.*plot',
            r'distribution.*plot',
            r'correlation.*heatmap',
            r'trend.*analysis.*plot',
        ]
        
        quality_score = sum(1 for pattern in quality_viz_patterns 
                          if re.search(pattern, code_snippet, re.IGNORECASE))
        score += (quality_score / len(quality_viz_patterns)) * 0.2
        
        return min(score, 1.0)
        
    except Exception:
        return 0.0


def get_enhanced_medical_scoring_reward(code_snippet: str) -> float:
    """
    Enhanced medical scoring reward with clinical validation logic.
    """
    try:
        score = 0.0
        total_checks = 15
        
        # Previous scoring checks (condensed)
        basic_scoring_score = get_medical_scoring_reward(code_snippet)
        score += basic_scoring_score * 0.4  # 40% from basic checks
        
        # Advanced scoring system patterns
        advanced_scoring = [
            r'composite.*score',  # Composite scoring systems
            r'weighted.*factor',  # Weighted scoring
            r'calibration.*curve',  # Score calibration
            r'validation.*cohort',  # External validation
            r'sensitivity.*specificity',  # Performance metrics
        ]
        
        advanced_score = sum(1 for pattern in advanced_scoring 
                           if re.search(pattern, code_snippet, re.IGNORECASE))
        score += (advanced_score / len(advanced_scoring)) * 0.2
        
        # Clinical decision support integration
        decision_support = [
            r'recommend.*based.*on.*score',
            r'alert.*high.*risk',
            r'escalation.*protocol',
            r'care.*pathway.*trigger',
            r'clinical.*action.*item',
        ]
        
        decision_score = sum(1 for pattern in decision_support 
                           if re.search(pattern, code_snippet, re.IGNORECASE))
        score += (decision_score / len(decision_support)) * 0.2
        
        # Score interpretation and communication
        interpretation_patterns = [
            r'patient.*friendly.*explanation',
            r'clinician.*summary',
            r'risk.*communication',
            r'shared.*decision.*making',
            r'confidence.*interval',
        ]
        
        interpretation_score = sum(1 for pattern in interpretation_patterns 
                                 if re.search(pattern, code_snippet, re.IGNORECASE))
        score += (interpretation_score / len(interpretation_patterns)) * 0.2
        
        return min(score, 1.0)
        
    except Exception:
        return 0.0


def get_comprehensive_medical_reward(
    code_snippet: str, 
    weights: Dict[str, float] = None,
    focus_area: str = 'balanced'
) -> float:
    """
    Comprehensive medical reward that adapts based on focus area.
    
    Args:
        code_snippet: Python code string
        weights: Dictionary of reward weights
        focus_area: 'visualization', 'scoring', 'safety', 'interop', or 'balanced'
        
    Returns:
        Float combined reward score (0.0 to 1.0)
    """
    if weights is None:
        if focus_area == 'visualization':
            weights = {
                'enhanced_visualization': 0.4,
                'enhanced_scoring': 0.1,
                'safety': 0.2,
                'clinical_logic': 0.1,
                'interoperability': 0.1,
                'code_quality_balance': 0.1
            }
        elif focus_area == 'scoring':
            weights = {
                'enhanced_visualization': 0.1,
                'enhanced_scoring': 0.4,
                'safety': 0.2,
                'clinical_logic': 0.2,
                'interoperability': 0.05,
                'code_quality_balance': 0.05
            }
        elif focus_area == 'safety':
            weights = {
                'enhanced_visualization': 0.1,
                'enhanced_scoring': 0.1,
                'safety': 0.4,
                'clinical_logic': 0.2,
                'interoperability': 0.1,
                'code_quality_balance': 0.1
            }
        elif focus_area == 'interop':
            weights = {
                'enhanced_visualization': 0.05,
                'enhanced_scoring': 0.05,
                'safety': 0.2,
                'clinical_logic': 0.1,
                'interoperability': 0.4,
                'code_quality_balance': 0.2
            }
        else:  # balanced
            weights = {
                'enhanced_visualization': 0.2,
                'enhanced_scoring': 0.2,
                'safety': 0.2,
                'clinical_logic': 0.15,
                'interoperability': 0.15,
                'code_quality_balance': 0.1
            }
    
    try:
        viz_reward = get_enhanced_medical_visualization_reward(code_snippet)
        scoring_reward = get_enhanced_medical_scoring_reward(code_snippet)
        safety_reward = get_medical_safety_reward(code_snippet)
        clinical_reward = get_clinical_logic_reward(code_snippet)
        interop_reward = get_medical_interoperability_reward(code_snippet)
        balance_reward = get_code_quality_medical_balance_reward(code_snippet)
        
        combined_reward = (
            weights['enhanced_visualization'] * viz_reward +
            weights['enhanced_scoring'] * scoring_reward +
            weights['safety'] * safety_reward +
            weights['clinical_logic'] * clinical_reward +
            weights['interoperability'] * interop_reward +
            weights['code_quality_balance'] * balance_reward
        )
        
        return min(combined_reward, 1.0)
        
    except Exception:
        return 0.0


def get_medical_safety_reward(code_snippet: str) -> float:
    """
    Reward for medical safety considerations and validation logic.
    
    Args:
        code_snippet: Python code string
        
    Returns:
        Float reward score (0.0 to 1.0)
    """
    try:
        score = 0.0
        total_checks = 12
        
        # Input validation and sanitization
        validation_patterns = [
            r'assert.*\d+.*<=.*<=.*\d+',  # Range assertions
            r'if.*not.*isinstance',  # Type checking
            r'validate.*input',  # Input validation functions
            r'check.*range',  # Range checking
            r'raise.*ValueError.*\(.*range',  # Proper error messages for ranges
        ]
        
        validation_score = sum(1 for pattern in validation_patterns 
                             if re.search(pattern, code_snippet, re.IGNORECASE))
        score += min(validation_score / 3, 1.0)
        
        # Medical value boundaries and limits
        medical_limits = [
            r'hr.*[<>]=?\s*\d+|heart_rate.*[<>]=?\s*\d+',  # HR limits
            r'bp.*[<>]=?\s*\d+|blood_pressure.*[<>]=?\s*\d+',  # BP limits
            r'age.*[<>]=?\s*\d+',  # Age limits
            r'weight.*[<>]=?\s*\d+',  # Weight limits
            r'temp.*[<>]=?\s*\d+|temperature.*[<>]=?\s*\d+',  # Temperature limits
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in medical_limits):
            score += 1.0
            
        # Critical alert and warning systems
        alert_patterns = [
            r'critical.*alert|alert.*critical',
            r'warning.*threshold|threshold.*warning',
            r'emergency.*value|value.*emergency',
            r'if.*critical.*\:',
            r'logging\.warning|logging\.error|logging\.critical',
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in alert_patterns):
            score += 1.0
            
        # Drug interaction and allergy checking logic
        safety_check_patterns = [
            r'drug.*interaction|interaction.*drug',
            r'allergy.*check|check.*allergy',
            r'contraindication',
            r'adverse.*effect|side.*effect',
            r'compatibility.*check',
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in safety_check_patterns):
            score += 1.0
            
        # Medical data privacy and HIPAA considerations
        privacy_patterns = [
            r'anonymize|de.*identify',
            r'encrypt.*data|data.*encrypt',
            r'patient.*id.*hash|hash.*patient',
            r'phi.*protect|protect.*phi',
            r'security.*check',
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in privacy_patterns):
            score += 1.0
            
        # Unit conversion and standardization
        conversion_patterns = [
            r'convert.*unit|unit.*convert',
            r'fahrenheit.*celsius|celsius.*fahrenheit',
            r'lbs.*kg|kg.*lbs',
            r'inches.*cm|cm.*inches',
            r'standardize.*unit',
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in conversion_patterns):
            score += 1.0
            
        # Medical coding standards (ICD, SNOMED, etc.)
        standards_patterns = [
            r'icd.*10|icd.*9',
            r'snomed.*ct|snomed',
            r'cpt.*code',
            r'loinc',
            r'fhir.*resource|fhir',
            r'hl7',
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in standards_patterns):
            score += 1.0
            
        # Temporal logic for medical data
        temporal_patterns = [
            r'time.*series.*valid',
            r'chronological.*order',
            r'before.*after.*check',
            r'date.*validation',
            r'temporal.*consistency',
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in temporal_patterns):
            score += 1.0
            
        # Medical calculation accuracy checks
        accuracy_patterns = [
            r'round.*\d+.*decimal',
            r'precision.*medical',
            r'significant.*digit',
            r'calculation.*verify',
            r'cross.*check.*result',
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in accuracy_patterns):
            score += 1.0
            
        # Audit trail and logging
        audit_patterns = [
            r'audit.*trail|trail.*audit',
            r'log.*medical.*event',
            r'track.*change',
            r'record.*modification',
            r'timestamp.*action',
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in audit_patterns):
            score += 1.0
            
        # Medical workflow validation
        workflow_patterns = [
            r'workflow.*step',
            r'clinical.*path',
            r'protocol.*follow',
            r'guideline.*check',
            r'evidence.*based',
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in workflow_patterns):
            score += 1.0
            
        # Quality assurance patterns
        qa_patterns = [
            r'quality.*check|check.*quality',
            r'validate.*result',
            r'verify.*calculation',
            r'double.*check',
            r'peer.*review',
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in qa_patterns):
            score += 1.0
            
        return min(score / total_checks, 1.0)
        
    except Exception:
        return 0.0


def get_clinical_logic_reward(code_snippet: str) -> float:
    """
    Reward for sophisticated clinical decision-making logic.
    
    Args:
        code_snippet: Python code string
        
    Returns:
        Float reward score (0.0 to 1.0)
    """
    try:
        score = 0.0
        total_checks = 10
        
        # Multi-criteria decision making
        decision_patterns = [
            r'if.*and.*and',  # Complex conditions
            r'elif.*or.*or',  # Multiple condition paths
            r'decision.*tree',
            r'criteria.*weight',
            r'score.*threshold',
        ]
        
        decision_score = sum(1 for pattern in decision_patterns 
                           if re.search(pattern, code_snippet, re.IGNORECASE))
        score += min(decision_score / 3, 1.0)
        
        # Evidence-based medicine patterns
        evidence_patterns = [
            r'evidence.*level|level.*evidence',
            r'clinical.*trial',
            r'meta.*analysis',
            r'systematic.*review',
            r'cochrane',
            r'pubmed.*id',
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in evidence_patterns):
            score += 1.0
            
        # Risk stratification logic
        risk_patterns = [
            r'risk.*stratif|stratif.*risk',
            r'low.*medium.*high.*risk',
            r'risk.*score.*categor',
            r'probability.*adverse',
            r'likelihood.*outcome',
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in risk_patterns):
            score += 1.0
            
        # Clinical reasoning chains
        reasoning_patterns = [
            r'differential.*diagnosis',
            r'rule.*out|exclude.*condition',
            r'confirm.*diagnosis',
            r'clinical.*impression',
            r'assessment.*plan',
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in reasoning_patterns):
            score += 1.0
            
        # Treatment recommendation logic
        treatment_patterns = [
            r'recommend.*treatment|treatment.*recommend',
            r'therapy.*select|select.*therapy',
            r'dose.*adjust|adjust.*dose',
            r'contraindic.*check',
            r'alternative.*treatment',
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in treatment_patterns):
            score += 1.0
            
        # Population health considerations
        population_patterns = [
            r'population.*specific',
            r'demographic.*factor',
            r'age.*group.*specific',
            r'pediatric.*adult.*geriatric',
            r'ethnicity.*consider',
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in population_patterns):
            score += 1.0
            
        # Comorbidity and complexity handling
        comorbidity_patterns = [
            r'comorbid|co.*morbid',
            r'multiple.*condition',
            r'complex.*case',
            r'interaction.*disease',
            r'polypharmacy',
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in comorbidity_patterns):
            score += 1.0
            
        # Follow-up and monitoring logic
        monitoring_patterns = [
            r'follow.*up.*schedule',
            r'monitor.*interval',
            r'repeat.*test',
            r'surveillance.*protocol',
            r'tracking.*progress',
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in monitoring_patterns):
            score += 1.0
            
        # Medical emergency detection
        emergency_patterns = [
            r'emergency.*detect|detect.*emergency',
            r'urgent.*care',
            r'critical.*value',
            r'immediate.*attention',
            r'red.*flag',
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in emergency_patterns):
            score += 1.0
            
        # Medical knowledge integration
        knowledge_patterns = [
            r'medical.*knowledge.*base',
            r'clinical.*guideline',
            r'protocol.*driven',
            r'best.*practice',
            r'standard.*care',
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in knowledge_patterns):
            score += 1.0
            
        return min(score / total_checks, 1.0)
        
    except Exception:
        return 0.0


def get_medical_interoperability_reward(code_snippet: str) -> float:
    """
    Reward for medical data interoperability and integration patterns.
    
    Args:
        code_snippet: Python code string
        
    Returns:
        Float reward score (0.0 to 1.0)
    """
    try:
        score = 0.0
        total_checks = 8
        
        # FHIR and HL7 integration
        fhir_patterns = [
            r'fhir.*resource',
            r'fhir.*client',
            r'bundle.*resource',
            r'patient.*resource',
            r'observation.*resource',
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in fhir_patterns):
            score += 1.0
            
        # API and web service integration
        api_patterns = [
            r'requests\.get|requests\.post',
            r'api\.call|call.*api',
            r'rest.*service',
            r'web.*service',
            r'json.*parse|parse.*json',
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in api_patterns):
            score += 1.0
            
        # Database integration with medical considerations
        db_patterns = [
            r'sql.*query.*patient',
            r'database.*connect',
            r'orm.*model',
            r'transaction.*medical',
            r'medical.*database',
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in db_patterns):
            score += 1.0
            
        # Medical data format handling
        format_patterns = [
            r'dicom.*parse|parse.*dicom',
            r'csv.*medical|medical.*csv',
            r'xml.*medical|medical.*xml',
            r'json.*fhir|fhir.*json',
            r'edi.*transaction',
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in format_patterns):
            score += 1.0
            
        # Error handling for integration
        integration_error_patterns = [
            r'connection.*error.*medical',
            r'timeout.*healthcare',
            r'retry.*medical.*api',
            r'fallback.*mechanism',
            r'graceful.*degradation',
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in integration_error_patterns):
            score += 1.0
            
        # Medical data synchronization
        sync_patterns = [
            r'sync.*medical.*data',
            r'real.*time.*update',
            r'cache.*medical',
            r'refresh.*patient.*data',
            r'consistency.*check',
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in sync_patterns):
            score += 1.0
            
        # Medical system messaging
        messaging_patterns = [
            r'hl7.*message',
            r'adt.*message',
            r'lab.*result.*message',
            r'order.*message',
            r'notification.*medical',
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in messaging_patterns):
            score += 1.0
            
        # Compliance and audit for integration
        compliance_patterns = [
            r'audit.*integration',
            r'compliance.*check',
            r'hipaa.*compliant',
            r'gdpr.*medical',
            r'regulation.*follow',
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in compliance_patterns):
            score += 1.0
            
        return min(score / total_checks, 1.0)
        
    except Exception:
        return 0.0


def get_code_quality_medical_balance_reward(code_snippet: str) -> float:
    """
    Reward that balances medical specificity with general coding best practices.
    
    Args:
        code_snippet: Python code string
        
    Returns:
        Float reward score (0.0 to 1.0)
    """
    try:
        score = 0.0
        total_checks = 10
        
        # General coding best practices (50% of score)
        general_practices = [
            r'def\s+\w+\s*\(',  # Function definitions
            r'class\s+\w+',  # Class definitions
            r'""".*"""',  # Docstrings
            r'#.*\w+',  # Comments
            r'try\s*:.*except',  # Error handling
        ]
        
        general_score = sum(1 for pattern in general_practices 
                          if re.search(pattern, code_snippet, re.DOTALL))
        score += (general_score / len(general_practices)) * 0.5
        
        # Medical domain application of good practices (50% of score)
        medical_good_practices = [
            r'def.*medical.*\(|def.*clinical.*\(',  # Medical function naming
            r'class.*Medical|class.*Clinical',  # Medical class naming
            r'""".*medical.*"""',  # Medical docstrings
            r'#.*medical|#.*clinical|#.*patient',  # Medical comments
            r'validate.*medical|medical.*validate',  # Medical validation
        ]
        
        medical_practice_score = sum(1 for pattern in medical_good_practices 
                                   if re.search(pattern, code_snippet, re.IGNORECASE | re.DOTALL))
        score += (medical_practice_score / len(medical_good_practices)) * 0.5
        
        return min(score, 1.0)
        
    except Exception:
        return 0.0


# Keep original functions for backward compatibility but mark as legacy
def get_medical_visualization_reward(code_snippet: str) -> float:
    """Legacy function - use get_enhanced_medical_visualization_reward instead."""
    try:
        score = 0.0
        total_checks = 10
        
        # Check for medical visualization libraries
        viz_libraries = ['matplotlib', 'plotly', 'seaborn', 'bokeh', 'altair']
        if any(lib in code_snippet.lower() for lib in viz_libraries):
            score += 1.0
            
        # Check for medical chart types
        medical_chart_patterns = [
            r'plt\.plot\s*\(',  # Line plots for trends
            r'plt\.scatter\s*\(',  # Scatter plots
            r'plt\.bar\s*\(',  # Bar charts
            r'plt\.subplot\s*\(',  # Multiple subplots
            r'fig\s*,\s*ax\s*=\s*plt\.subplots',  # Proper subplot setup
            r'\.line\s*\(',  # Plotly line charts
            r'\.scatter\s*\(',  # Plotly scatter
        ]
        
        chart_matches = sum(1 for pattern in medical_chart_patterns 
                           if re.search(pattern, code_snippet, re.IGNORECASE))
        score += min(chart_matches / 3, 1.0)  # Up to 1.0 for multiple chart types
        
        # Check for medical terminology and units
        medical_terms = [
            'hr', 'heart_rate', 'bp', 'blood_pressure', 'spo2', 'oxygen', 
            'temperature', 'temp', 'respiratory_rate', 'rr', 'pulse',
            'systolic', 'diastolic', 'medication', 'dose', 'vital',
            'percentile', 'growth', 'height', 'weight', 'bmi'
        ]
        
        medical_units = [
            'bpm', 'mmhg', '%', '°c', '°f', 'celsius', 'fahrenheit',
            'mg', 'ml', 'kg', 'cm', 'inches', 'lbs'
        ]
        
        term_matches = sum(1 for term in medical_terms 
                          if term in code_snippet.lower())
        unit_matches = sum(1 for unit in medical_units 
                          if unit in code_snippet.lower())
        
        score += min((term_matches + unit_matches) / 5, 1.0)
        
        # Check for proper axis labeling
        label_patterns = [
            r'xlabel\s*\(',
            r'ylabel\s*\(',
            r'title\s*\(',
            r'legend\s*\(',
            r'xaxis.*title',
            r'yaxis.*title'
        ]
        
        label_matches = sum(1 for pattern in label_patterns 
                           if re.search(pattern, code_snippet, re.IGNORECASE))
        score += min(label_matches / 3, 1.0)
        
        # Check for time series handling
        time_patterns = [
            r'datetime', r'pd\.to_datetime', r'strptime', r'timestamp',
            r'time_series', r'resample', r'freq=', r'date_range'
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in time_patterns):
            score += 1.0
            
        # Check for medical data preprocessing
        preprocessing_patterns = [
            r'fillna\s*\(', r'dropna\s*\(', r'interpolate\s*\(',
            r'rolling\s*\(', r'ewm\s*\(', r'smooth',
            r'outlier', r'clip\s*\(', r'between\s*\('
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in preprocessing_patterns):
            score += 1.0
            
        # Check for proper medical color schemes
        color_patterns = [
            r'color\s*=', r'cmap\s*=', r'palette\s*=',
            r'red|blue|green', r'#[0-9a-fA-F]{6}'
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in color_patterns):
            score += 1.0
            
        # Check for medical reference lines/ranges
        reference_patterns = [
            r'axhline\s*\(', r'axvline\s*\(',
            r'fill_between\s*\(', r'normal.*range',
            r'threshold', r'critical.*value'
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in reference_patterns):
            score += 1.0
            
        # Check for error handling in medical context
        error_patterns = [
            r'try\s*:', r'except', r'ValueError', r'TypeError',
            r'assert.*range', r'if.*valid', r'check.*input'
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in error_patterns):
            score += 1.0
            
        # Check for documentation/comments about medical context
        doc_patterns = [
            r'#.*normal.*range', r'#.*vital', r'#.*medical',
            r'""".*medical.*"""', r"'''.*medical.*'''"
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE | re.DOTALL) for pattern in doc_patterns):
            score += 1.0
            
        return min(score / total_checks, 1.0)
        
    except Exception:
        return 0.0


def get_medical_scoring_reward(code_snippet: str) -> float:
    """Legacy function - use get_enhanced_medical_scoring_reward instead."""
    try:
        score = 0.0
        total_checks = 10
        
        # Check for known medical scoring systems
        scoring_systems = [
            'apache', 'nihss', 'glasgow', 'gcs', 'cha2ds2vasc', 'chads2',
            'wells', 'meld', 'child_pugh', 'sofa', 'saps', 'qsofa',
            'curb65', 'psi', 'grace', 'timi', 'framingham'
        ]
        
        scoring_matches = sum(1 for system in scoring_systems 
                             if system in code_snippet.lower())
        score += min(scoring_matches / 2, 1.0)
        
        # Check for proper scoring calculation patterns
        calc_patterns = [
            r'score\s*\+=', r'points\s*\+=', r'total.*score',
            r'calculate.*score', r'compute.*score', r'sum\s*\(',
            r'np\.sum\s*\(', r'score\s*=.*\+', r'weight.*factor'
        ]
        
        calc_matches = sum(1 for pattern in calc_patterns 
                          if re.search(pattern, code_snippet, re.IGNORECASE))
        score += min(calc_matches / 3, 1.0)
        
        # Check for medical parameter validation
        validation_patterns = [
            r'if.*age.*between', r'if.*\d+.*<=.*<=.*\d+',
            r'range\s*\(.*,.*\)', r'min\s*=.*max\s*=',
            r'valid.*range', r'normal.*values', r'check.*bounds'
        ]
        
        validation_matches = sum(1 for pattern in validation_patterns 
                               if re.search(pattern, code_snippet, re.IGNORECASE))
        score += min(validation_matches / 2, 1.0)
        
        # Check for medical conditions/criteria handling
        condition_patterns = [
            r'if.*condition', r'elif.*criteria', r'case.*when',
            r'severity.*level', r'risk.*category', r'class.*level'
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in condition_patterns):
            score += 1.0
            
        # Check for proper missing value handling
        missing_patterns = [
            r'is.*none', r'nan', r'null', r'missing',
            r'pd\.isna', r'np\.isnan', r'fillna', r'default.*value'
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in missing_patterns):
            score += 1.0
            
        # Check for medical units and ranges
        medical_ranges = [
            r'\d+.*mmhg', r'\d+.*bpm', r'\d+.*mg/dl',
            r'\d+.*years', r'age.*\d+', r'normal.*\d+.*\d+'
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in medical_ranges):
            score += 1.0
            
        # Check for result interpretation
        interpretation_patterns = [
            r'low.*risk', r'high.*risk', r'moderate.*risk',
            r'severe', r'mild', r'normal', r'abnormal',
            r'interpret.*score', r'category.*score'
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in interpretation_patterns):
            score += 1.0
            
        # Check for documentation of medical logic
        doc_patterns = [
            r'#.*score.*point', r'#.*criteria', r'#.*medical',
            r'""".*scoring.*"""', r"'''.*algorithm.*'''"
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE | re.DOTALL) for pattern in doc_patterns):
            score += 1.0
            
        # Check for error handling and edge cases
        error_patterns = [
            r'try\s*:', r'except.*Error', r'raise.*Error',
            r'assert.*valid', r'check.*input', r'validate.*parameter'
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in error_patterns):
            score += 1.0
            
        # Check for return type and structure
        return_patterns = [
            r'return.*score', r'return.*dict', r'return.*tuple',
            r'return.*\{.*\}', r'return.*\(.*,.*\)'
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in return_patterns):
            score += 1.0
            
        return min(score / total_checks, 1.0)
        
    except Exception:
        return 0.0


def get_medical_data_handling_reward(code_snippet: str) -> float:
    """
    Calculate reward for proper medical data handling practices.
    
    Args:
        code_snippet: Python code string
        
    Returns:
        Float reward score (0.0 to 1.0)
    """
    try:
        score = 0.0
        total_checks = 8
        
        # Check for medical data libraries
        medical_libs = ['pandas', 'numpy', 'scipy', 'scikit-learn', 'datetime']
        lib_matches = sum(1 for lib in medical_libs if lib in code_snippet)
        score += min(lib_matches / 3, 1.0)
        
        # Check for proper data validation
        validation_patterns = [
            r'pd\.read_csv', r'pd\.DataFrame', r'data\.head\(\)',
            r'data\.info\(\)', r'data\.describe\(\)', r'data\.shape'
        ]
        
        if any(re.search(pattern, code_snippet) for pattern in validation_patterns):
            score += 1.0
            
        # Check for medical data cleaning
        cleaning_patterns = [
            r'drop_duplicates', r'remove_outliers', r'clean.*data',
            r'standardize', r'normalize', r'convert.*units'
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in cleaning_patterns):
            score += 1.0
            
        # Check for time-based operations
        time_patterns = [
            r'groupby.*time', r'resample', r'rolling.*window',
            r'shift\(\)', r'lag', r'lead', r'time.*delta'
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in time_patterns):
            score += 1.0
            
        # Check for medical aggregations
        agg_patterns = [
            r'mean\(\)', r'median\(\)', r'std\(\)', r'min\(\)', r'max\(\)',
            r'quantile', r'percentile', r'aggregate'
        ]
        
        agg_matches = sum(1 for pattern in agg_patterns 
                         if re.search(pattern, code_snippet))
        score += min(agg_matches / 3, 1.0)
        
        # Check for medical filtering
        filter_patterns = [
            r'query\s*\(', r'loc\[', r'iloc\[', r'where\s*\(',
            r'filter.*condition', r'select.*criteria'
        ]
        
        if any(re.search(pattern, code_snippet) for pattern in filter_patterns):
            score += 1.0
            
        # Check for proper medical calculations
        calc_patterns = [
            r'calculate.*rate', r'compute.*average', r'trend.*analysis',
            r'correlation', r'regression', r'statistical.*test'
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in calc_patterns):
            score += 1.0
            
        # Check for result formatting
        format_patterns = [
            r'round\s*\(', r'format\s*\(', r'\.2f', r'precision',
            r'to_string', r'to_dict', r'to_json'
        ]
        
        if any(re.search(pattern, code_snippet) for pattern in format_patterns):
            score += 1.0
            
        return min(score / total_checks, 1.0)
        
    except Exception:
        return 0.0


def get_medical_domain_knowledge_reward(code_snippet: str) -> float:
    """
    Calculate reward based on medical domain knowledge demonstrated in code.
    
    Args:
        code_snippet: Python code string
        
    Returns:
        Float reward score (0.0 to 1.0)
    """
    try:
        score = 0.0
        
        # Medical terminology usage (weighted scoring)
        medical_terms = {
            'vital_signs': ['heart_rate', 'blood_pressure', 'temperature', 'respiratory_rate', 'oxygen_saturation'],
            'medications': ['dosage', 'medication', 'drug', 'prescription', 'administration'],
            'conditions': ['diagnosis', 'symptom', 'disease', 'disorder', 'syndrome'],
            'procedures': ['treatment', 'therapy', 'intervention', 'procedure', 'protocol'],
            'measurements': ['lab_value', 'test_result', 'biomarker', 'indicator', 'parameter']
        }
        
        category_scores = []
        for category, terms in medical_terms.items():
            term_count = sum(1 for term in terms if term in code_snippet.lower())
            category_scores.append(min(term_count / len(terms), 1.0))
        
        score += np.mean(category_scores) * 0.4  # 40% weight
        
        # Medical constants and reference values
        reference_values = [
            r'normal.*range', r'reference.*value', r'threshold.*\d+',
            r'critical.*limit', r'target.*range', r'baseline.*value'
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in reference_values):
            score += 0.2  # 20% weight
            
        # Clinical workflow understanding
        workflow_patterns = [
            r'patient.*flow', r'clinical.*pathway', r'care.*plan',
            r'assessment.*monitoring', r'follow.*up', r'discharge.*criteria'
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in workflow_patterns):
            score += 0.2  # 20% weight
            
        # Safety and validation considerations
        safety_patterns = [
            r'safety.*check', r'validate.*input', r'alert.*condition',
            r'warning.*threshold', r'emergency.*value', r'critical.*alert'
        ]
        
        if any(re.search(pattern, code_snippet, re.IGNORECASE) for pattern in safety_patterns):
            score += 0.2  # 20% weight
            
        return min(score, 1.0)
        
    except Exception:
        return 0.0


def get_combined_medical_reward(code_snippet: str, weights: Dict[str, float] = None) -> float:
    """
    Legacy combined reward function for backward compatibility.
    Use get_comprehensive_medical_reward for enhanced functionality.
    """
    if weights is None:
        weights = {
            'visualization': 0.3,
            'scoring': 0.3,
            'data_handling': 0.2,
            'domain_knowledge': 0.2
        }
    
    try:
        viz_reward = get_medical_visualization_reward(code_snippet)
        scoring_reward = get_medical_scoring_reward(code_snippet)
        data_reward = get_medical_data_handling_reward(code_snippet)
        domain_reward = get_medical_domain_knowledge_reward(code_snippet)
        
        combined_reward = (
            weights['visualization'] * viz_reward +
            weights['scoring'] * scoring_reward +
            weights['data_handling'] * data_reward +
            weights['domain_knowledge'] * domain_reward
        )
        
        return min(combined_reward, 1.0)
        
    except Exception:
        return 0.0 