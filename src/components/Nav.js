import Link from "next/link";

export default function Nav() {
  return (
    <nav className="flex gap-4 border-b border-zinc-200 px-6 py-4 dark:border-zinc-800">
      <Link href="/">Home</Link>
      <Link href="/projects">Projects</Link>
    </nav>
  );
}
