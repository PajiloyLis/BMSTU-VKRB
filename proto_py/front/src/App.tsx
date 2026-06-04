import { ParseForm } from "./components/ParseForm";
import { ParseResult } from "./components/ParseResult";
import { useParser } from "./hooks/useParser";

export function App() {
  const { result, isLoading, error, submit } = useParser();

  return (
    <main className="app">
      <section className="hero">
        <h1>Разбор предложения</h1>
        <p>Введите предложение и запустите разбор.</p>
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
