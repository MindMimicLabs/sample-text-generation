require(dplyr)
require(readtext)
require(tokenizers)

load_corpus <- function(corpus_path) {

  # load the documents
  documents <-
    list.files(
      path = corpus_path,
      recursive = TRUE,
      pattern = "\\.txt",
      full.names = TRUE)

  max_sentence_length = 0
  words_in_documents <- vector(mode = 'list', length = length(documents))
  for(i in 1:length(documents)) {
    document <- documents[i]
    lines <-
      document %>%
      readtext() %>%
      tokenize_lines(simplify = T)
    words_in_document <- vector(mode = 'list', length = length(lines))
    for(j in 1:length(lines)) {
      #  EOS marks the end of a sentence for when we flatten the list later
      words_in_document[[j]] <- c(lines[j] %>% tokenize_words(simplify = T), 'EOS')
    }
    t1 <- max(sapply(words_in_document, length))
    max_sentence_length <- max(max_sentence_length, t1)
    words_in_documents[[i]] <- words_in_document %>% unlist()
    print(sprintf("document # %d: words: %d. max sentence length: %d", i, length(words_in_documents[[i]]), t1))
  }  
  
  vocab <-
    words_in_documents %>%
    unlist(recursive = T) %>%
    unique()
  print(sprintf("unique words: %d", length(vocab)))
  
  list(documents = words_in_documents, vocab = vocab, max_sentence_length = max_sentence_length)
}
