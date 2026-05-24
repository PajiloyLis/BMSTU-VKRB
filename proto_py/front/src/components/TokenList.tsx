interface TokenListProps {
  tokens: string[];
}

export function TokenList({ tokens }: TokenListProps) {
  if (tokens.length === 0) {
    return null;
  }

  return (
    <div className="tokens">
      {tokens.map((token, index) => (
        <span className="token" key={`${token}-${index}`}>
          {token}
        </span>
      ))}
    </div>
  );
}
