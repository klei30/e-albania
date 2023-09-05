import { OpenAIChat } from 'langchain/llms';
import { LLMChain, ChatVectorDBQAChain, loadQAChain } from 'langchain/chains';
import { PineconeStore } from 'langchain/vectorstores';
import { PromptTemplate } from 'langchain/prompts';
import { CallbackManager } from 'langchain/callbacks';

const CONDENSE_PROMPT = PromptTemplate.fromTemplate(`
Given the following conversation and a follow up question, craft in Albanian language the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`);


const QA_PROMPT = PromptTemplate.fromTemplate(`
Ju jeni një asistent që ndihmon përdoruesit me informacion rreth shërbimeve të E-albania në gjuhën shqipe. 
Ju duhet të ofroni përgjigje të saktë, të detajuara dhe faktike, duke u bazuar vetëm në dokumentet e përfshira.
Është e rëndësishme të ndiqni striktësisht informacionin nga dokumentet origjinale.

Nëse ju pyesin për diçka që nuk është në dokumentet e përfshira, refuzojeni të përgjigjeni me korrektësi. 
Përgjigjet tuaja duhet të jenë miqësore dhe të orientuara ndaj përdoruesit.
Nese ju pershendesin, pershendeti edhe ti miqesisht

{context}
Pyetja: {question}
Përgjigjja në HTML markdown: `);



export const makeChain = (
  vectorstore: PineconeStore,
  onTokenStream?: (token: string) => void,
) => {
  const questionGenerator = new LLMChain({
      llm: new OpenAIChat({ temperature: 0.3} ),
      prompt: CONDENSE_PROMPT,
    });
    const docChain = loadQAChain(
      new OpenAIChat({
        modelName: 'gpt-3.5-turbo', //change this to gpt-4 if you have access    
        maxTokens:1000,
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
    k: 1, //number of source documents to return
  });
};
