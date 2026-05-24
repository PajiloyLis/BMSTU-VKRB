import { ParseForm } from "./components/ParseForm";
import { ParseResult } from "./components/ParseResult";
import { useParser } from "./hooks/useParser";

export function App() {
  const { result, isLoading, error, submit } = useParser();

  return (
    <main className="app">
      <section className="hero">
        <p className="eyebrow">CYK + русская морфология</p>
        <h1>Разбор предложения</h1>
        <p>
          Введите предложение, получите токены, скобочную запись, DOT/PNG дерева и JSON,
          сохранённые по публичным URL.
        </p>
      </section>

      <div className="layout">
        <ParseForm isLoading={isLoading} onSubmit={submit} />
        <div>
          {error && <section className="panel error">{error}</section>}
          <ParseResult result={result} />
        </div>
      </div>
    </main>
  );
}
