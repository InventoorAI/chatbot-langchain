import {
    Message as VercelChatMessage,
    StreamingTextResponse,
    createStreamDataTransformer
} from 'ai';
import { ChatMistralAI } from '@langchain/mistralai';
import { PromptTemplate } from '@langchain/core/prompts';

import { JSONLoader } from "langchain/document_loaders/fs/json";
import { RunnableSequence } from '@langchain/core/runnables';
import { formatDocumentsAsString } from 'langchain/util/document';
import { HttpResponseOutputParser } from "langchain/output_parsers";

//load json in format
const loader = new JSONLoader(
    "src/data/data.json",
    ["/name", "/quantity", "/description", "/location", "/weight", "/id"],
);

export const dynamic = 'force-dynamic'

const formatMessage = (message: VercelChatMessage) => {
    return `${message.role}: ${message.content}`;
};

const TEMPLATE = `
Answer the user's questions based only on the following context.
If user is giving a command, execute it.
If the answer is not in the context, reply politely that you do not have that information available.:
==============================
Context: {context}
==============================
Current conversation: {chat_history}

user: {question}
assistant:`;

export async function POST(req: Request) {
    try {
        const { messages } = await req.json();

        const formattedPreviousMessages = messages.slice(0, -1).map(formatMessage);

        const currentMessageContent = messages[messages.length - 1].content;

        const docs = await loader.load();
        const formattedDocs = formatDocumentsAsString(docs);

        const prompt = PromptTemplate.fromTemplate(TEMPLATE);

        const model = new ChatMistralAI({
            apiKey: process.env.MISTRAL_API_KEY!,
            model: 'mistral-small',
            temperature: 0,
            streaming: true,
        });

        const parser = new HttpResponseOutputParser();

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

        const stream = await chain.stream({
            chat_history: formattedPreviousMessages.join('\n'),
            question: currentMessageContent,
        });

        return new StreamingTextResponse(
            stream.pipeThrough(createStreamDataTransformer()),
        );
    } catch (e: any) {
        return Response.json({ error: e.message }, { status: e.status ?? 500 });
    }
}