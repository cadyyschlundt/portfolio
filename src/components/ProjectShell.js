export default function ProjectShell({
  title,
  whatAndWhy,
  limitations,
  whatILearned,
  children,
}) {
  return (
    <main className="mx-auto max-w-2xl px-6 py-12">
      <h1 className="text-2xl font-bold">{title}</h1>

      <section className="mt-8">
        <h2 className="text-lg font-semibold">What and why</h2>
        <p className="mt-2">{whatAndWhy}</p>
      </section>

      <section className="mt-8">
        <h2 className="text-lg font-semibold">Interactive piece</h2>
        <div className="mt-2">{children}</div>
      </section>

      <section className="mt-8">
        <h2 className="text-lg font-semibold">Limitations</h2>
        <p className="mt-2">{limitations}</p>
      </section>

      <section className="mt-8">
        <h2 className="text-lg font-semibold">What I learned</h2>
        <p className="mt-2">{whatILearned}</p>
      </section>
    </main>
  );
}
