import ProjectShell from "@/components/ProjectShell";
import CostCurve from "@/components/german-credit/CostCurve";
// Local JSON import - Next reads this file at build time, so there is no fetch and no loading state.
import results from "./results.json";

export default function GermanCreditPage() {
  return (
    <ProjectShell
      title="German Credit Default Risk"
      whatAndWhy={
        <>
          <p>
            This project is a binary classifier trained on the UCI German Credit dataset, predicting whether a loan applicant is a good or bad credit risk. But the classifier itself isn't really the point. The two ways it can be wrong cost different amounts, and I built the whole project around letting that cost asymmetry drive the modeling decisions instead of just optimizing a generic metric like accuracy.
          </p>
          <p className="mt-4">
            The two errors are not equally bad in the real world. Denying a loan to someone who would have repaid it costs the lender their interest income, a bounded and fairly modest loss. Approving a loan to someone who then defaults can cost the lender the entire principal, a much larger loss. That gap between a missed profit and a lost principal is why I weighted a false negative, approving a bad borrower, five times as heavily as a false positive, denying a good one, instead of treating the two mistakes as the same.
          </p>
          <p className="mt-4">
            Judged on accuracy, this model loses to the simplest possible rule: approving every applicant gets 70% right just by picking the majority class, while this model only gets 62.67% right on the test set. If I had judged this on accuracy alone, I would have thrown it out. But approve-everyone isn't the right baseline for what the two errors actually cost. Once you're counting cost instead of accuracy, reject-everyone becomes the cheaper of the two trivial rules. Measured against that baseline, this model cuts average cost per applicant by 29.5%. Accuracy doesn't just treat the two errors as equal in the abstract here. It would have actively misled me into throwing out a model that wins on the metric that actually matches how a lender loses money.
          </p>
        </>
      }
      limitations={
        <>
          <p>
            This is a portfolio project built to demonstrate method, not a production model.
          </p>
          <p className="mt-4">
            The dataset is the German Credit data: old, from a different country's lending market, and small at 1000 rows. Nothing I conclude here would transfer to a real modern lending book without redoing this work on real data.
          </p>
          <p className="mt-4">
            The 5-to-1 weighting of a false negative against a false positive is a number I chose for this project, not one derived from real loss data. A real lender would estimate this ratio from their own loss-given-default and lost-interest figures. The whole result rests on this number, and I picked it because this is a learning exercise.
          </p>
          <p className="mt-4">
            The model itself is modest. Its ROC AUC is only 0.7991, which is okay but not strong. The value of this project is the rigor around a modest model, not the model's raw predictive strength.
          </p>
          <p className="mt-4">
            Claims about subgroups get shaky when the groups are small. My under-25 fairness number rests on only 88 people, so I lean on the comparison rather than the exact figure.
          </p>
          <p className="mt-4">
            Under-25 applicants are wrongly denied at about 69%, roughly double the rate for over-40 applicants at 33%.
          </p>
          <p className="mt-4">
            I tested whether dropping the protected attributes removes this disparity, and the two attributes behaved differently. Dropping the sex feature mostly removed the sex disparity, because the model was using it fairly directly. Dropping age barely changed the age disparity, because proxy features like employment and credit history correlate with age and reconstruct it. So removing a protected attribute only fixes bias when there are no proxies for it.
          </p>
        </>
      }
      whatILearned={
        <>
          <p>
            This project showed me the difference between what a model can do on its own and what needs a person to see. The model just optimizes whatever objective it's given. Most of the actual judgment here lived in the decisions I made, not in the model.
          </p>
          <p className="mt-4">
            I chose the 5-to-1 cost ratio. I tuned the decision threshold to match that cost structure. I computed the trivial baselines on the cost metric instead of accuracy. I ran the fairness audit. The model executed all of that, but none of those choices came from it.
          </p>
          <p className="mt-4">
            The model also reproduced a bias that was already sitting in the data, and it gave no signal that it was doing so. It took a deliberate audit to catch it. The model does what you tell it, including things you didn't realize you were telling it.
          </p>
          <p className="mt-4">
            The rigor in this project lived in the human decisions around the model, not in the model itself.
          </p>
        </>
      }
    >
      <p>
        Test-set average cost is {results.test_result.avg_cost}, a{" "}
        {results.test_result.pct_reduction_vs_baseline}% reduction from the
        reject-everyone baseline of {results.baseline.avg_cost}.
      </p>
      <p>
        The decision threshold was tuned to {results.threshold.chosen}, versus
        the default of {results.threshold.default}.
      </p>
      <p>
        Choosing that threshold cut cost {results.threshold.pct_improvement_vs_default}%
        compared to the default threshold.
      </p>
      <CostCurve points={results.cost_curve.points} />
    </ProjectShell>
  );
}
