# A Principled Guide to Multiclass Classification Metrics

## The Foundation: What Are You Actually Trying to Measure?

Before selecting any metric, you need to answer a fundamental question that shapes everything else: what constitutes success in your specific problem? This question has multiple dimensions that often conflict with each other.

**Overall correctness versus class-specific performance.** Some classes might be more prevalent in your data, and accuracy can be misleading when class distributions are skewed. In your home activity prediction scenario, if "idle" dominates your sensor data while "cooking" is rare, a classifier that simply predicts "idle" most of the time might achieve high accuracy while completely failing at the task you actually care about.

**Error symmetry versus asymmetry.** Do all misclassifications carry equal weight? In medical diagnosis, confusing "healthy" with "minor illness" is fundamentally different from confusing "healthy" with "life-threatening condition." Cost-sensitive learning addresses scenarios where different kinds of errors have different consequences, and the objective becomes minimizing total misclassification cost rather than error rate. For your sensor data, misclassifying "sleeping" as "sitting" might be tolerable, while confusing "falling" with any other activity could be critical.

**Discrimination versus calibration.** Expected calibration error quantifies how well predicted output probabilities match actual probabilities of the ground truth distribution. A model might achieve excellent ranking of predictions (good discrimination) but produce poorly calibrated confidence scores. If you need reliable probability estimates for downstream decision-making, calibration matters; if you only need rank-ordered predictions, you can focus on discrimination metrics.

## Class Imbalance: The Core Structural Challenge

Class imbalance fundamentally changes which metrics are meaningful. Understanding the averaging strategies—macro, micro, and weighted—is essential, but more important is understanding what each reveals and conceals about your model's behavior.

**Macro-averaging: The egalitarian view.** Macro-averaging calculates each class's performance metric and takes the arithmetic mean across all classes, giving equal weight to each class regardless of the number of instances. This approach treats a rare class with 10 examples the same as a common class with 1000 examples. When is this appropriate?

Use macro-averaging when all classes have equal importance in your application, regardless of their frequency. If your home activity classifier needs to detect rare but important activities (like medical emergencies) just as well as common ones (like sitting), macro-averaging makes poor performance on rare classes immediately visible. When you have an imbalanced dataset and want to ensure each class equally contributes to the final evaluation, macro-averaging is useful.

The critical limitation: macro-averaging can disguise poor performance in a critical minority class when the overall number of classes is large, since each class's contribution is diluted. With 20 activity classes, one completely failing class only contributes 5% to the macro-averaged score.

**Micro-averaging: The utilitarian view.** Micro-averaging aggregates the counts of true positives, false positives, and false negatives across all classes and calculates the performance metric based on total counts, giving equal weight to each instance regardless of class label. Remarkably, micro-averaged precision, recall, and F1 are all equal to overall accuracy in multiclass single-label classification.

Use micro-averaging when you care about overall prediction quality across all instances, and when your class distribution in deployment will match your evaluation data. Micro-averaging can be appropriate when you want to account for the total number of misclassifications in the dataset. However, micro-averaging can overemphasize the performance of the majority class, especially when it dominates the dataset.

**Weighted averaging: The pragmatic compromise.** Weighted averaging adjusts for class imbalance by weighting each class's contribution according to its frequency in the dataset. This reflects the reality that you'll encounter frequent classes more often in deployment.

The key consideration: weighted averaging assumes your deployment distribution matches your evaluation distribution. If your sensor deployment will see different activity distributions than your training data (perhaps because test homes have different occupancy patterns), weighted averaging based on training frequencies becomes misleading.

## The Confusion Matrix: Your Source of Truth

Every metric you'll use derives from the confusion matrix. For multiclass problems, this becomes a K×K matrix where entry (i,j) represents instances of true class i predicted as class j. The diagonal contains correct predictions; everything else represents specific types of errors.

**What the confusion matrix reveals.** Unlike aggregate metrics, the confusion matrix shows you the specific error patterns your model makes. You might discover that your model never confuses "cooking" with "sleeping" but frequently confuses "eating" with "sitting." This structural information about error patterns is lost when you reduce everything to a single number.


---

*Only partially relelvant now*
For your LOOCV scenario, examining confusion matrices across different test homes reveals whether certain activity confusions are consistent across homes or home-specific. If Home A's model confuses activities X and Y, but Home B's doesn't, this suggests feature engineering opportunities or fundamental differences in how those activities manifest across homes.

---

**Per-class metrics versus aggregation.** Calculating metrics by category is useful when you want to evaluate the performance of a particular class and know how well the classifier can distinguish this class from others. Report per-class precision, recall, and F1 when:

- You have specific classes of high importance that need monitoring
- Your audience needs to understand class-specific trade-offs
- You're diagnosing model failure modes
- Class count is manageable (reporting 50 per-class metrics is overwhelming)

Calculating precision and recall for each class can result in a large number of performance metrics that can be challenging to interpret together. When you have many classes, aggregated metrics become necessary for tractable evaluation, but you should still examine per-class performance for your most important classes.

## Beyond Accuracy: The Classic Metrics

**Accuracy: Deceptively simple.** Accuracy is the proportion of correct predictions. For a balanced dataset, an accuracy of 100%/k where k is the number of classes represents random guessing performance. Accuracy has exactly one appropriate use case: balanced classes where all errors are equally costly. The moment either assumption breaks, accuracy becomes misleading.

**Precision and recall: Class-specific guarantees.** Precision answers "when I predict class X, how often am I right?" Recall answers "when the true class is X, how often do I catch it?" These represent different operational guarantees.

High precision means few false alarms for that class—important when acting on a prediction is costly. High recall means few missed instances—important when missing an instance is costly. The F1 score is their harmonic mean, balancing both concerns. F1 score is especially useful when you need to balance precision and recall and there's a trade-off between them.

For your activity recognition: if you're triggering an alert when detecting "emergency" activities, you need high precision (few false alarms). If you're logging all instances of "medication time" for health monitoring, you need high recall (can't miss any).

## Advanced Considerations: When Simple Metrics Fail

**The Matthews Correlation Coefficient: The balanced alternative.** The Matthews correlation coefficient produces a high score only if the prediction obtained good results in all four confusion matrix categories (true positives, false negatives, true negatives, false positives), proportionally both to the size of positive elements and the size of negative elements. 

MCC has a natural multiclass extension and ranges from -1 to +1, where +1 is perfect classification and -1 is asymptotically reached in extreme misclassification cases. Unlike accuracy and F1 score which can show overoptimistic results on imbalanced datasets, MCC provides a more reliable statistical rate.

Use MCC when you need a single balanced metric that naturally accounts for class imbalance and treats all confusion matrix elements fairly. The limitation: MCC is a correlation coefficient, which makes it less interpretable than precision/recall for stakeholders who need to understand specific error types.

**Cohen's Kappa: Agreement beyond chance.** Cohen's kappa measures inter-annotator agreement, expressing the level of agreement between predicted and actual classifications beyond what would be expected by random chance. Kappa adjusts for the base rate of each class, making it valuable when you want to know how much better than random guessing your model performs.

**Cost-sensitive metrics: When errors have prices.** In many real-world applications, the costs of different types of mistakes are unequal. You can define a cost matrix C where C[i,j] represents the cost of predicting class j when the true class is i.

In the multi-class case, if costs depend only on the true label (not the predicted label), theoretical thresholds can be derived to adjust prediction probabilities. However, rescaling approaches work well when costs are consistent, but directly applying them to multi-class problems with inconsistent costs may not be effective.

For your LOOCV scenario: if certain activity misclassifications have real consequences (confusing "medication time" with "general activity" means a missed dose), define explicit costs and evaluate using mean misclassification cost rather than accuracy.

**Calibration metrics: Trusting the probabilities.** Expected Calibration Error (ECE) measures how well predicted output probabilities match actual probabilities by binning predictions and comparing average confidence in each bin to actual accuracy. Three variants exist:
- L1 norm (ECE): weighted average of absolute differences
- L2 norm (RMSCE): root mean square differences  
- Max norm (MCE): maximum difference across bins

Calibration matters when you'll use predicted probabilities for decision-making rather than just hard classifications. In your sensor data scenario, if you're building a system that adjusts confidence thresholds based on context, well-calibrated probabilities are essential. If you're just classifying activities for logging, discrimination metrics matter more than calibration.

## What to Leave Out: Strategic Omissions

**Don't report metrics that don't match your objective.** If you care about rare class detection, don't lead with overall accuracy. If costs are asymmetric, don't report symmetric metrics like balanced accuracy without also reporting cost-aware metrics.

**Don't aggregate when aggregation hides critical information.** If you have 3-5 classes of high interest and 15 filler classes, reporting only macro-averaged F1 obscures whether your model succeeds at what matters. Report per-class metrics for critical classes alongside aggregated metrics for completeness.

**Don't ignore the null model.** Always establish baseline performance. For balanced classes, random guessing achieves 1/k accuracy. For imbalanced classes, always-predict-majority achieves frequency of majority class. Your model must substantially beat these baselines on the metrics that matter.

**Don't confuse ranking metrics with threshold metrics.** ROC-AUC evaluates ranking quality across all possible thresholds. Precision, recall, and F1 evaluate performance at a specific threshold. ROC AUC provides no information about precision and negative predictive value at your operating point. Report both if both matter; don't substitute one for the other.

**Don't report calibration if you don't use probabilities.** If your deployment pipeline uses only hard class predictions (argmax), calibration metrics are irrelevant. Focus on discrimination metrics instead.

## The LOOCV Consideration

Your leave-one-out cross-validation setup across homes introduces specific considerations. You're training on H-1 homes and testing on the held-out home, which creates a natural distribution shift between train and test.

**Variance across folds matters.** Don't just report average metrics across LOOCV folds. Report variance or confidence intervals. High variance suggests your model's performance is house-dependent, indicating you may need home-specific adaptation or more robust features.

**Per-home error analysis.** Some homes might consistently show poor performance on specific activities. This isn't a failure—it's actionable information about sensor placement, activity patterns, or the need for personalization.

**The macro vs. weighted trade-off in LOOCV.** Weighted averaging based on the test home's class distribution makes sense for evaluating performance on that specific home. Macro-averaging tells you about average performance across all activities regardless of how often they occur in any particular home. Both perspectives are valid; report both and explain what each means.

## A Decision Framework

Here's a systematic approach to selecting metrics:

**Step 1: Define success operationally.** What does a good prediction look like in deployment? What errors are tolerable versus critical?

**Step 2: Characterize your data.** Balanced or imbalanced? Few or many classes? Reliable labels or noisy ground truth?

**Step 3: Match metrics to operational needs.**
- Need overall quality across all predictions? → Micro-averaged metrics or accuracy (if balanced)
- Need equal performance across all classes? → Macro-averaged metrics
- Need to reflect deployment distribution? → Weighted metrics
- Have asymmetric error costs? → Cost-sensitive metrics
- Need probability estimates? → Add calibration metrics
- Need interpretable guarantees? → Per-class precision/recall for key classes
- Need a single balanced number? → MCC or Cohen's Kappa

**Step 4: Report complementary metrics.** No single metric captures everything. A minimal reporting set includes:
- One aggregate discrimination metric (macro/micro/weighted F1 based on your needs)
- Per-class metrics for your most important classes
- Confusion matrix (or subset for key classes if too large)
- Baseline comparison
- For LOOCV: variance/confidence intervals across folds

**Step 5: Validate metric choice.** Does improving your chosen metric actually make the system better at its real task? If optimizing macro-F1 leads to a model that works worse in practice, you've chosen the wrong metric.

## Principles Over Rules

The refusal to give simple rules of thumb is intentional. Your multiclass classification problem has structure that emerges from:
- The physical nature of activities and their sensor signatures
- The distribution of activities across homes  
- The consequences of different error types
- Your deployment constraints and requirements

These specifics determine which metrics reveal truth and which obscure it. A metric that's essential for medical diagnosis might be irrelevant for activity logging. Macro-averaging might be exactly right for your rare-activity-detection problem or completely wrong if deployment distribution differs from test.

The principled approach is to understand what each metric measures, what it assumes, and what it hides—then choose metrics that align with your actual definition of success while honestly representing your model's behavior.