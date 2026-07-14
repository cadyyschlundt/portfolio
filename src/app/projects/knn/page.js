import ProjectShell from "@/components/ProjectShell";

export default function KnnPage() {
  return (
    <ProjectShell
      title="k-Nearest Neighbors Playground"
      whatAndWhy="This project explores how the k-nearest neighbors algorithm classifies a point by looking at the labels of the points closest to it."
      limitations="This is a placeholder page with no working algorithm yet, so it doesn't handle real data, ties, or higher dimensions."
      whatILearned="Building this page established the reusable shell and route structure every future project page will follow."
    >
      <div className="border border-zinc-300 p-4 dark:border-zinc-700">
        interactive piece goes here
      </div>
    </ProjectShell>
  );
}
