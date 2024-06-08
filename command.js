require('dotenv').config();
import { ChatMistralAI } from "@langchain/mistralai";
import { PromptTemplate } from "@langchain/core/prompts";
import { JSONLoader } from "langchain/document_loaders/fs/json";
import { RunnableSequence } from "@langchain/core/runnables";
import { formatDocumentsAsString } from "langchain/util/document";
import { StringOutputParser } from "@langchain/core/output_parsers";
import express from 'express';

const app = express();
app.use(express.json());

const loader = new JSONLoader("src/data/data.json", [
  "/name",
  "/quantity",
  "/description",
  "/location",
  "/weight",
  "/id",
]);

const formatMessage = (message) => {
  return `${message.role}: ${message.content}`;
};

const TEMPLATE = `
Answer the user's questions based only on the following context.
If user is giving a command, execute it.
==============================
Context: {context}
==============================
Current conversation: {chat_history}

user: {question}
assistant:`;

async function handlePostRequest(req, res) {
  try {
    const { messages } = req.body;

    const formattedPreviousMessages = messages.slice(0, -1).map(formatMessage);
    const currentMessageContent = messages[messages.length - 1].content;
    const docs = await loader.load();
    const formattedDocs = formatDocumentsAsString(docs);
    const prompt = PromptTemplate.fromTemplate(TEMPLATE);

    const model = new ChatMistralAI({
      apiKey: process.env.MISTRAL_API_KEY,
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

    const stream = await chain.invoke({
      chat_history: formattedPreviousMessages.join("\n"),
      question: currentMessageContent,
    });

    return res.json({ messages: stream });
  } catch (e) {
    console.log("error ==> ", e);
    res.status(e.status ?? 500).json({ error: e.message });
  }
}

app.post("/api", handlePostRequest);

const PORT = process.env.PORT;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
