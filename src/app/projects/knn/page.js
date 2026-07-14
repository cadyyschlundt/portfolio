import ProjectShell from "@/components/ProjectShell";
import PointCanvas from "@/components/knn/PointCanvas";

export default function KnnPage() {
  return (
    <ProjectShell
      title="k-Nearest Neighbors Playground"
      whatAndWhy="This project explores how the k-nearest neighbors algorithm classifies a point by looking at the labels of the points closest to it."
      limitations="This phase only places labeled points on a canvas. There is no k, no classifier, and no decision boundary yet."
      whatILearned="Building this piece established how points live in state as the source of truth, with the canvas as a pure rendering of that state, plus how to handle canvas coordinates correctly on high-DPI screens."
    >
      <PointCanvas />
    </ProjectShell>
  );
}
