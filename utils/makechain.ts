import { OpenAIChat } from 'langchain/llms';
import { LLMChain, ChatVectorDBQAChain, loadQAChain } from 'langchain/chains';
import { PineconeStore } from 'langchain/vectorstores';
import { PromptTemplate } from 'langchain/prompts';
import { CallbackManager } from 'langchain/callbacks';

const CONDENSE_PROMPT =
  PromptTemplate.fromTemplate(`Given the following conversation and a follow up question, rephrase in Albanian language the follow up question to be a standalone question.DO not produce new and fake information AND DO NOT SEND LINKS  

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`);

const QA_PROMPT = PromptTemplate.fromTemplate(`
  Ju jeni një sistem që asiston përdoruesit rreth shërbimeve të E-albania në gjuhën shqipe. Roli juaj është të ofroni informacion të saktë, të detajuar, dhe faktik, duke u bazuar në të dhënat e suportuara. 
  Kur ju pershendesin, pershendeti edhe ti.

  Ju duhet të ndjekni striktësisht të dhënat e bazuara në dokumentet burimore, dhe çdo devijim do të penalizohet. Veçanërisht për URL-të, gjithmonë duhet të jepni linke të sakta dhe të paprekura nga burimi origjinal. 

  Përgjigjet tuaja duhet të jenë miqësore dhe të orientuara drejt objektivave dhe rrethanave të përdoruesit, duke suportuar një përvojë të këndshme. 

  Nëse merrni pyetje që nuk janë të lidhura me E-Albania dhe materialin e suportuar, në mënyrë miqësore refuzoni tu përgjigjeni.

  Ju jeni të lidhur me parimet e mësipërme dhe çdo shkelje e tyre është e papranueshme.

  Tani, keni për detyrë të përgjigjeni pyetjes së përdoruesit, duke përdorur markdown, duke gjeneruar një listë që përfaqëson hapat që përdoruesi duhet të ndjekë, dhe duke përfshirë URL-në themelore në përgjigjen tuaj.
 Mbaj mend dhe mos dergo Kurr Linke ose hiperlinke pervecse kur eshte vetem https://e-albania.al/
  Konteksti: {context}
  Pyetja: {question}
  Përgjigjja (në HTML markdown): `);



export const makeChain = (
  vectorstore: PineconeStore,
  onTokenStream?: (token: string) => void,
) => {
  const questionGenerator = new LLMChain({
      llm: new OpenAIChat({ temperature: 0} ),
      prompt: CONDENSE_PROMPT,
    });
    const docChain = loadQAChain(
      new OpenAIChat({
        modelName: 'gpt-3.5-turbo-16k',//change this to older versions (e.g. gpt-3.5-turbo) or (gpt-4) 
        maxTokens:4000,
        topP:0.8,
        presencePenalty:-0.5,
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
