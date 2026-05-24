import { FormEvent, useState } from "react";

interface ParseFormProps {
  isLoading: boolean;
  onSubmit: (sentence: string, maxTrees: number) => void;
}

export function ParseForm({ isLoading, onSubmit }: ParseFormProps) {
  const [sentence, setSentence] = useState("Я люблю чай.");
  const [maxTrees, setMaxTrees] = useState(3);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onSubmit(sentence, maxTrees);
  }

  return (
    <form className="panel parse-form" onSubmit={handleSubmit}>
      <label htmlFor="sentence">Предложение</label>
      <textarea
        id="sentence"
        value={sentence}
        onChange={(event) => setSentence(event.target.value)}
        placeholder="Введите русское предложение"
        rows={5}
      />

      <div className="form-row">
        <label htmlFor="maxTrees">Деревьев вернуть</label>
        <input
          id="maxTrees"
          type="number"
          min={1}
          max={20}
          value={maxTrees}
          onChange={(event) => setMaxTrees(Number(event.target.value))}
        />
      </div>

      <button type="submit" disabled={isLoading || sentence.trim().length === 0}>
        {isLoading ? "Разбираю..." : "Разобрать"}
      </button>
    </form>
  );
}
