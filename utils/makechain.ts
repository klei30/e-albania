import { OpenAIChat } from 'langchain/llms';
import { LLMChain, ChatVectorDBQAChain, loadQAChain } from 'langchain/chains';
import { PineconeStore } from 'langchain/vectorstores';
import { PromptTemplate } from 'langchain/prompts';
import { CallbackManager } from 'langchain/callbacks';

const CONDENSE_PROMPT =
  PromptTemplate.fromTemplate(`Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`);

const QA_PROMPT = PromptTemplate.fromTemplate(
  `Dua që të veprosh si një asistent qe shkruan shqip për navigimin e shërbimeve të e-Albania.
  Ti do lexosh nga një dokument PDF në e-Albania dhe do të ofrosh përgjigje strikte të sakta. Nëse nuk e di përgjigjen, thjesht thuaj 
  se nuk e di, në vend se të shpikësh një. Thekso se je programuar për të përgjigjur pyetjeve që kanë lidhje me 
  context dhe përgjigju me mirësjellje pyetjeve që nuk kanë lidhje me context. Jepu tonin e miqësor, 
  i mirësjellshëm dhe shpjegoi gjërat në detaje. Asistoi gjithmonë hap pas hapi në përdorimin 
  e shërbimeve të e-Albania. Për më tepër, përdor lidhjet për t'u referuar ndaj shërbimeve të ndryshme.
  Dergo edhe cdo link ose hyprlinks sipas context
 {context}
 
 Question: {question}
 Helpful answer in markdown:`,
);

export const makeChain = (
  vectorstore: PineconeStore,
  onTokenStream?: (token: string) => void,
) => {
  const questionGenerator = new LLMChain({
    llm: new OpenAIChat({ temperature: 1 }),
    prompt: CONDENSE_PROMPT,
  });
  const docChain = loadQAChain(
    new OpenAIChat({
      temperature: 0.7,
      modelName: 'gpt-3.5-turbo-16k',//change this to older versions (e.g. gpt-3.5-turbo) or (gpt-4) 
      maxTokens:2000 ,
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
