import { ChatMistralAI } from "@langchain/mistralai";
import { PromptTemplate } from "@langchain/core/prompts";
import { JSONLoader } from "langchain/document_loaders/fs/json";
import { RunnableSequence } from "@langchain/core/runnables";
import { formatDocumentsAsString } from "langchain/util/document";
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
            
      const loader = new JSONLoader(table, [
        "/name",
        "/quantity",
        "/description",
        "/width",
        "/height",
        "/depth",
        "/site"
      ]);
      const docs = await loader.load();
      const formattedDocs = formatDocumentsAsString(docs);
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
  
      return output
    } catch (e) {
      console.log("error ==> ", e);
    }
  }

app.post("/api", handlePostRequest);

const PORT = 4000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
