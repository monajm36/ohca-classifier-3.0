# OHCA Annotation Guidelines v3.0 - Enhanced Methodology

## Overview

This document provides comprehensive guidelines for manually annotating discharge notes to identify Out-of-Hospital Cardiac Arrest (OHCA) cases using the enhanced v3.0 methodology that addresses data scientist feedback about bias, data leakage, and evaluation issues.

## Enhanced v3.0 Methodology

This annotation guide supports the improved v3.0 training methodology with:

- **Larger sample sizes**: 800 training + 200 validation cases (vs 264 in legacy)
- **Separate annotation files**: Training and validation annotated independently  
- **Patient-level data splits**: Prevents data leakage in model training
- **Optimal threshold finding**: Validation set used for threshold optimization
- **Unbiased evaluation**: Independent test set reserved for final assessment

### Important: You Will Receive TWO Separate Files

1. **Training file** (800 cases) - Used for model training
2. **Validation file** (200 cases) - Used for threshold optimization

**Do NOT annotate the test file** - this is reserved for final unbiased evaluation.

This separation prevents bias and improves model reliability compared to legacy single-file annotation.

## Definition of OHCA

**Out-of-Hospital Cardiac Arrest (OHCA)** is a cardiac arrest that occurs **outside** a healthcare facility (home, workplace, public spaces, etc.) and is the **primary reason** for the current hospital admission.

## Annotation Labels

- **1 = OHCA**: Cardiac arrest that occurred outside a healthcare facility and is the primary reason for admission
- **0 = Non-OHCA**: Everything else (including transfers, historical arrests, non-arrest conditions)

## Include as OHCA (Label = 1)

### Must Meet ALL Criteria:
- Cardiac arrest occurred outside any healthcare facility
- Arrest is the primary reason for this hospital admission
- Not a transfer case from another hospital
- Current episode, not historical

### Examples:
- Cardiac arrest at home with family CPR
- Cardiac arrest at work with coworker CPR  
- Cardiac arrest in public spaces with bystander CPR
- Cardiac arrest witnessed by non-medical personnel
- Out-of-hospital arrest with EMS resuscitation

## Exclude as Non-OHCA (Label = 0)

### In-Hospital Cardiac Arrests
- Cardiac arrest occurring within any healthcare facility
- Code blue called on hospital ward, ICU, ED
- Arrest during surgery or medical procedures
- Arrest in any hospital unit or clinical area

### Transfer Cases (ALL EXCLUDED)
- **All transfer cases are excluded**, including:
- Patients transferred from outside hospitals for OHCA
- Patients transferred for cardiac catheterization after OHCA
- Inter-facility transfers for any reason
- **Rule**: If admission note mentions "transfer" → Label = 0

### Historical Cardiac Arrests
- Patients with history of cardiac arrest not related to current admission
- Previous arrest episodes mentioned in past medical history
- Cardiac arrest that occurred days/weeks/months before current admission

### Non-Arrest Conditions
- Chest pain without cardiac arrest
- Heart attack (MI) without cardiac arrest  
- Shortness of breath, dyspnea
- Syncope/fainting without arrest
- Near-syncope or pre-syncope
- Any condition where cardiac arrest did not occur

### Trauma-Related Arrests
- Cardiac arrest secondary to trauma
- Motor vehicle accident with arrest
- Falls, drowning, or other trauma-induced arrest

## Decision Tree

```
1. Did a cardiac arrest occur during this admission?
   └── NO → Label = 0 (Stop here)
   └── YES → Continue to 2

2. Did the arrest happen OUTSIDE a healthcare facility?
   └── NO (in-hospital) → Label = 0 (Stop here)
   └── YES → Continue to 3

3. Is this a transfer case from another hospital?
   └── YES → Label = 0 (Stop here)
   └── NO → Continue to 4

4. Is OHCA the PRIMARY reason for this admission?
   └── NO → Label = 0 (Stop here)
   └── YES → Label = 1 (OHCA case)
```

## Confidence Scale

Rate your confidence in the annotation (1-5 scale):

- **5 = Very Confident**: Clear, unambiguous case with strong evidence
- **4 = Confident**: Strong evidence, only minor uncertainty
- **3 = Moderately Confident**: Some ambiguity but leaning toward decision  
- **2 = Uncertain**: Significant ambiguity, difficult case to classify
- **1 = Very Uncertain**: Unclear, conflicting information, needs expert review

## Detailed Examples with Rationale

### Example 1: Clear OHCA at Home (Label = 1)
**Text**: "Patient found down at home by spouse at 0800, immediate CPR initiated by family, EMS arrived and achieved ROSC after 15 minutes of resuscitation."

**Rationale**: 
- ✅ Cardiac arrest occurred (found down, CPR, ROSC)
- ✅ Outside hospital (at home)
- ✅ Not a transfer case
- ✅ Primary reason for admission
- **Label**: 1, **Confidence**: 5

### Example 2: In-Hospital Arrest (Label = 0)
**Text**: "Patient admitted for pneumonia management, developed cardiac arrest on hospital day 3 while on medical ward."

**Rationale**:
- ✅ Cardiac arrest occurred
- ❌ Inside hospital (medical ward)
- **Label**: 0, **Confidence**: 5

### Example 3: Transfer for OHCA - EXCLUDED (Label = 0)
**Text**: "Transfer from community hospital. Patient had witnessed cardiac arrest at home yesterday, family CPR given, transferred here for advanced cardiac care."

**Rationale**:
- ✅ Cardiac arrest occurred outside hospital originally
- ❌ This is a transfer case (key exclusion)
- **Label**: 0, **Confidence**: 5

### Example 4: Historical Arrest (Label = 0)
**Text**: "Patient with history of cardiac arrest 1 year ago, now presents with acute chest pain and shortness of breath."

**Rationale**:
- ❌ No current cardiac arrest
- ❌ Historical arrest, not current admission
- **Label**: 0, **Confidence**: 5

### Example 5: MI Without Arrest (Label = 0)
**Text**: "Chief complaint: Chest pain. Patient presents with acute STEMI, underwent emergency PCI with good result."

**Rationale**:
- ❌ No cardiac arrest occurred (MI is not arrest)
- **Label**: 0, **Confidence**: 5

### Example 6: Workplace Cardiac Arrest (Label = 1)
**Text**: "Patient collapsed at workplace during meeting, witnessed by coworkers who initiated immediate CPR, EMS transported with sustained ROSC."

**Rationale**:
- ✅ Cardiac arrest occurred (collapsed, CPR needed)
- ✅ Outside hospital (at workplace)
- ✅ Not a transfer case
- ✅ Primary reason for admission
- **Label**: 1, **Confidence**: 5

### Example 7: Public Place Arrest (Label = 1)
**Text**: "Witnessed collapse at shopping mall, bystander CPR initiated immediately, public AED used with successful defibrillation."

**Rationale**:
- ✅ Cardiac arrest occurred (collapse, CPR, defibrillation)
- ✅ Outside hospital (shopping mall = public place)
- ✅ Not a transfer case
- ✅ Primary reason for admission
- **Label**: 1, **Confidence**: 5

### Example 8: Syncope Without Arrest (Label = 0)
**Text**: "Patient experienced brief loss of consciousness at home, family called EMS, patient awake and alert on arrival."

**Rationale**:
- ❌ No cardiac arrest (brief LOC ≠ arrest, no CPR mentioned)
- **Label**: 0, **Confidence**: 4

## Quality Control (Enhanced for v3.0)

### For Larger Annotation Batches
- **Take regular breaks**: Annotate in sessions of 50-100 cases to maintain quality
- **Track consistency**: Keep notes on borderline cases for reference across files
- **Regular calibration**: Review difficult cases with team periodically
- **Progress monitoring**: Check inter-annotator agreement on sample subsets
- **Cross-file consistency**: Ensure consistent decision-making between training and validation files

### Annotation Workflow (v3.0)
1. **Start with training file** (800 cases) - complete fully before moving to validation
2. **Complete validation file** (200 cases) - use same criteria and decision patterns
3. **Do NOT annotate test file** (reserved for final evaluation)
4. **Maintain consistency** between both annotation files
5. **Document difficult cases** in both files for reference

### Before Submitting
1. **Double-check** each decision against the four-step criteria
2. **Review** all cases with confidence < 3
3. **Add detailed notes** for any unusual or borderline cases
4. **Ensure consistency** between training and validation annotations
5. **Verify completeness** of both annotation files

### Notes Field Usage
Use the notes field to document:
- Reasoning for difficult decisions
- Key phrases that influenced the decision
- Uncertainties or missing information  
- Questions for expert review
- Cross-references to similar cases in the other file

## Efficient Annotation for Large Samples

### Time Management
- Plan 2-4 minutes per case on average
- Schedule annotation sessions with adequate time and breaks
- Use consistent workspace and eliminate distractions
- Set daily targets (e.g., 100 cases per day)

### Consistency Strategies
- **Reference cases**: Mark clear examples of each category for future reference
- **Decision log**: Keep running notes on reasoning for edge cases
- **Regular review**: Periodically review previous decisions for consistency
- **Pattern recognition**: Develop consistent approaches to common scenarios

### Annotation Tips
- Read the entire discharge summary, not just chief complaint
- Look for key phrases: "found down", "unresponsive", "CPR", "ROSC", "AED"
- Pay attention to location indicators: "at home", "at work", "in public" vs "in hospital"
- Watch for transfer language: "transferred from", "outside hospital", "referring facility"

## Common Mistakes to Avoid

1. **Don't include in-hospital arrests** (any arrest in healthcare facility)
2. **Don't include ANY transfer cases** (even transfers specifically for OHCA)
3. **Don't include historical arrests** from previous admissions
4. **Don't include conditions where no cardiac arrest occurred** (MI, chest pain, syncope)
5. **Don't assume arrest occurred** without clear evidence (CPR, ROSC, defibrillation)
6. **Don't ignore transfer status** - all transfers are excluded regardless of original arrest location

## Difficult Cases and Edge Cases

### Transfer Cases (Always Exclude)
- **All transfer cases are excluded (Label = 0)**
- This includes transfers specifically for OHCA management
- Look for phrases: "transferred from", "outside hospital", "referring facility"
- Focus on transfer status, not the original location of arrest

### Incomplete Information
- Base decision on available information in the discharge summary
- Use lower confidence score (1-2) for unclear cases
- Add detailed notes about missing information
- When in doubt about arrest occurrence, lean toward exclusion (Label = 0)

### Multiple Conditions
- Focus on whether cardiac arrest actually occurred
- Look for specific evidence: CPR, ROSC, defibrillation, "found down"
- If only chest pain, shortness of breath, or syncope mentioned → Label = 0

### Trauma-Related Cases
- Cardiac arrest secondary to trauma is typically excluded
- Focus on the primary cause: if trauma caused arrest → Label = 0
- If spontaneous arrest with incidental trauma → case-by-case decision

## Integration with OHCA Classifier v3.0

These guidelines support the enhanced training pipeline:

- **Data preparation**: Use `ohca-prepare labeled` to format your annotation files
- **After annotation**: Use `complete_annotation_and_train_v3()` for model training  
- **Automatic optimization**: Model will find optimal threshold using validation annotations
- **Enhanced inference**: Clinical decision support available in inference results
- **Quality tracking**: Model metadata includes annotation quality metrics

See training examples in the package documentation for complete workflow.

## Support and Questions

If you encounter cases that don't fit these guidelines:

1. **Make your best judgment** based on the four-step decision tree
2. **Use a lower confidence score** (1-2) to flag uncertainty
3. **Document detailed reasoning** in the notes field
4. **Flag for expert review** if needed
5. **Maintain consistency** across both annotation files

### Key Decision Points for Difficult Cases:
- **When unsure if arrest occurred**: Look for CPR, ROSC, defibrillation evidence
- **When unsure about location**: "At home", "at work", "in public" = outside hospital
- **When unsure about transfer status**: Any mention of transfer = exclude
- **When unsure about primary reason**: Focus on chief complaint and admission reason

## Key Principles Summary

1. **OHCA must occur outside healthcare facilities** (home, work, public spaces)
2. **All transfer cases are excluded** regardless of original arrest location  
3. **Only current cardiac arrests**, not historical episodes
4. **Clear evidence of arrest required**: CPR, ROSC, defibrillation, "found down"
5. **When uncertain, exclude (Label = 0)** and document reasons
6. **Maintain consistency** between training and validation annotations
7. **Patient-level methodology** prevents data leakage in model training

## v3.0 Methodology Benefits

This enhanced annotation approach supports:
- **Larger, more robust training datasets**
- **Unbiased model evaluation**  
- **Optimal threshold finding**
- **Better generalization to new datasets**
- **Enhanced clinical decision support**
- **Reproducible, peer-reviewed methodology**

Remember: Your careful annotation work directly impacts model quality and clinical utility. Consistency and attention to detail are essential for training reliable AI systems for healthcare.
