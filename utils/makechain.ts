import { OpenAIChat } from 'langchain/llms';
import { LLMChain, ChatVectorDBQAChain, loadQAChain } from 'langchain/chains';
import { PineconeStore } from 'langchain/vectorstores';
import { PromptTemplate } from 'langchain/prompts';
import { CallbackManager } from 'langchain/callbacks';

const CONDENSE_PROMPT =
PromptTemplate.fromTemplate(`Dhuruar bisedën e mëposhtme dhe një pyetje ndjekëse, riartikuloni pyetjen ndjekëse në gjuhën shqipe dhe dialektet e saj, për të qenë një pyetje e pavarur Sigurohuni qe mos te japesh KURR URL ose Links dhe mos e permend kete fakt tek pergjigja resposne. Mos Shpifs.

Chat History:
{chat_history}
Ndjekja e pyetjes: {question}
Standalone question:`);

const QA_PROMPT = PromptTemplate.fromTemplate(
  `Roli juaj është të shërbeni si një asistent ne gjuhen shqipe që lehtëson përdorimin e shërbimeve të e-Albania https://e-albania.al/, duke ofruar
   përgjigje në gjuhën shqipe. Do të merrni informacion nga PDF-të brenda e-Albania, duke siguruar saktësinë e 
   përgjigjeve. Thekso se je programuar për të përgjigjur pyetjeve që kanë lidhje me 
   context dhe përgjigju me mirësjellje pyetjeve që nuk kanë lidhje me context. Jepu tonin e miqësor, 
   i mirësjellshëm dhe shpjegoi gjërat në detaje. Asistoi gjithmonë hap pas hapi në përdorimin 
   e shërbimeve të e-Albania. Per te perdorur cdo sherbim, Shtypni butonin "Perdor" për të paraqitur aplikimin tuaj.

 {context}
 
 Question: {question}
 Helpful answer in markdown:`,
);

export const makeChain = (
  vectorstore: PineconeStore,
  onTokenStream?: (token: string) => void,
) => {
  const questionGenerator = new LLMChain({
    llm: new OpenAIChat({ temperature: 0.8  }),
    prompt: CONDENSE_PROMPT,
  });
  const docChain = loadQAChain(
    new OpenAIChat({
      modelName: 'gpt-3.5-turbo-16k',//change this to older versions (e.g. gpt-3.5-turbo) or (gpt-4) 
      maxTokens:2000 ,
      topP:0.3,
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
