import { ChatMistralAI } from "@langchain/mistralai";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import express from 'express';

const app = express();
app.use(express.json());

const formatMessage = (message) => {
  return `${message.role}: ${message.content}`;
};

const TEMPLATE = `
Answer the user's questions based only on the following context.
If the answer is not in the context, reply politely that you do not have that information available.:
==============================
Context: {context}
==============================
Current conversation: {chat_history}

user: {question}
assistant:`;

export async function chat(messages, table) {
    try {
      const formattedPreviousMessages = messages.slice(0, -1).map(formatMessage);
      const currentMessageContent = messages[messages.length - 1].content;
      const formattedDocs = JSON.stringify(table, null);
      const prompt = PromptTemplate.fromTemplate(TEMPLATE);
  
      const model = new ChatMistralAI({
        apiKey: "PdqHnloyBQVxi2vAR6eN2c4bOjQjnKLp",
        model: "mistral-small",
        temperature: 0,
        streaming: false,
      });
  
      const parser = new StringOutputParser();
  
      const chain = RunnableSequence.from([
        {
          question: (input) => input.question,
          chat_history: (input) => input.chat_history,
          context: () => formattedDocs,
        },
        prompt,
        model,
        parser,
      ]);
  
      const output = await chain.invoke({
        chat_history: formattedPreviousMessages.join("\n"),
        question: currentMessageContent,
      });
  
      return output;
    } catch (e) {
      console.log("error ==> ", e);
    }
  }

async function handlePostRequest(req, res) {
  const { messages } = req.body;

  const latestMessage = messages[messages.length - 1].content

  try {
    const response = await fetch('http://192.168.145.49:8000/parse', {method: "POST", headers: {"Content-Type": "application/json"},body:JSON.stringify({"prompt": latestMessage})})
    const data  = await response.json()

    if (data.valid){
      //send to   
      return res.json({
        "message": "On it!"
      })
    }
  } finally {
    const table = [{name: "A", quantity: 12, site: "Site A"}, {name: "B", quantity: 110, site: "Site B"}]
    const response = await chat(messages, table);

    return res.json({
      "message": response
    })
  }

}

app.post("/api", handlePostRequest);

const PORT = 4000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
