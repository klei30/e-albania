import { OpenAIChat } from 'langchain/llms';
import { LLMChain, ChatVectorDBQAChain, loadQAChain } from 'langchain/chains';
import { PineconeStore } from 'langchain/vectorstores';
import { PromptTemplate } from 'langchain/prompts';
import { CallbackManager } from 'langchain/callbacks';

const CONDENSE_PROMPT =
  PromptTemplate.fromTemplate(`Given the following conversation and a follow up question, rephrase in Albanian language the follow up question to be a standalone question.DO not produce new and fake information

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`);

const QA_PROMPT = PromptTemplate.fromTemplate(
  `Roli juaj është të shërbeni si një asistent ne gjuhen shqipe që lehtëson përdorimin e shërbimeve të e-Albania https://e-albania.al/.
   Referohu gjithmone te dokumenti origjinal,
   mos dil kurr jashte context. Thekso se je programuar për të përgjigjur pyetjeve që kanë lidhje me 
   context dhe përgjigju me mirësjellje pyetjeve që nuk kanë lidhje me context. Jepu tonin e miqësor, 
   i mirësjellshëm dhe shpjegoi gjërat në detaje. Asistoi gjithmonë hap pas hapi në përdorimin 
   e shërbimeve të e-Albania. Sigurohuni qe mos te japesh KURR URL ose Links dhe mos e permend kete fakt tek pergjigja resposne.
   mos prodho informacion te ri.
 {context}
 
 Question: {question}
 Helpful answer in markdown:`,
);

export const makeChain = (
  vectorstore: PineconeStore,
  onTokenStream?: (token: string) => void,
) => {
  const questionGenerator = new LLMChain({
      llm: new OpenAIChat({ temperature: 0.1}),
      prompt: CONDENSE_PROMPT,
    });
    const docChain = loadQAChain(
      new OpenAIChat({
        modelName: 'gpt-3.5-turbo-16k',//change this to older versions (e.g. gpt-3.5-turbo) or (gpt-4) 
        maxTokens:1500,
        topP:0.5,
      streaming: Boolean(onTokenStream),
      callbackManager: onTokenStream
        ? CallbackManager.fromHandlers({
            async handleLLMNewToken(token) {
              onTokenStream(token);
              console.log(token);
            },
          })
        : undefined,
    }),
    { prompt: QA_PROMPT },
  );

  return new ChatVectorDBQAChain({
    vectorstore,
    combineDocumentsChain: docChain,
    questionGeneratorChain: questionGenerator,
    returnSourceDocuments: true,
    k: 5, //number of source documents to return
  });
};
