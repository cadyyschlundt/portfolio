import Link from "next/link";

export default function ProjectsIndex() {
  return (
    <main className="mx-auto max-w-2xl px-6 py-12">
      <h1 className="text-2xl font-bold">Projects</h1>
      <ul className="mt-8">
        <li>
          <Link href="/projects/knn" className="underline">
            k-Nearest Neighbors Playground
          </Link>
        </li>
        <li>
          <Link href="/projects/german-credit" className="underline">
            German Credit Default Risk
          </Link>
        </li>
      </ul>
    </main>
  );
}
