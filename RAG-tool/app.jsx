import React, { useState } from "react";


export default function App() {
    const [question, setQuestion] = useState("");
    const [answer, setAnswer] = useState("");


    const ask = async () => {
        const res = await fetch("http:localhost:8000/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({question }),
        });
        const data = await res.json();
        setAnswer(data.answer);
    };

    return (
        <dev style={{ padding: 20 }}>
            <h1>Product Catalog QA</h1>
            <input
              style={{ width: "60%" }}
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask about a product..."
            />
            <button onClick={ask}> Ask</button>
            <div style={{ marginTop: 20 }}>
                <strong>Answer:</strong>
                <p>{answer}</p>
            </div>
        </dev>
    );
}