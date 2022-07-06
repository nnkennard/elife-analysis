import json
import pandas as pd


HTML_TEMPLATE = """
<HTML>
   <head>
      <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma@0.9.4/css/bulma.min.css">
      <link rel="stylesheet" href="css/dataTables.bulma.min.css">
      <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css">
   </head>
   <body>
      <div class="container">
         <div class="columns">
            <div class="column is-half">
               REVIEW_TABLE
            </div>
            <div class="column is-half">
               REBUTTAL_TABLE
            </div>
         </div>
         <div class="columns">
            <div class="column">
               OVERALL_TABLE
            </div>
         </div>
      </div>
      <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.1.1/jquery.min.js"></script>
      <script src="https://cdn.datatables.net/1.12.0/js/jquery.dataTables.min.js"></script>
      <script src="js/dataTables.bulma.min.js"></script>
      <script type="text/javascript">
         $(".fancytable").DataTable();
      </script>
   </body>
</HTML>
"""


def write_review_to_html(review, directory):
  review_table = pd.DataFrame.from_dict(
    [{"review_sentence": sent} for sent in review["review_sentences"]]).to_html().replace(
    '<table border="1" class="dataframe">', '<table border="1" class="table">').replace("\n", "")
  overall_table_dicts = []
  rebuttal_table_dicts = []

  for reb_i, reb_sent in enumerate(review["rebuttal_sentences"]):
    reb_sent_values = {
      "sbert": [],
      "jaccard": []
    }
    for rev_i, rev_sent in enumerate(review["review_sentences"]):
      sbert_val = review["result"]["sbert"][reb_i][rev_i]
      jaccard_val = review["result"]["jaccard"][reb_i][rev_i]
      reb_sent_values["sbert"].append(sbert_val)
      reb_sent_values["jaccard"].append(jaccard_val)
      overall_table_dicts.append({
        "review_sentence": rev_sent,
        "rebuttal_sentence": reb_sent,
        "sbert": sbert_val,
        "jaccard": jaccard_val,
      })
    rebuttal_table_dicts.append({
      "rebuttal_sentence": reb_sent,
      "sbert_mean": sum(reb_sent_values['sbert'])/len(reb_sent_values['sbert']),
      "jaccard_mean": sum(reb_sent_values['jaccard'])/len(reb_sent_values['jaccard'])
    })
  overall_table = pd.DataFrame.from_dict(overall_table_dicts).to_html().replace('<table border="1" class="dataframe">', '<table border="1" class="fancytable">').replace("\n", "")
  rebuttal_table = pd.DataFrame.from_dict(rebuttal_table_dicts).to_html().replace('<table border="1" class="dataframe">', '<table border="1" class="fancytable">').replace("\n", "")
  with open(f'{directory}/{review["review_id"]}.html', 'w') as f:
    f.write(HTML_TEMPLATE.replace("\n", "").replace("REVIEW_TABLE", review_table).replace("OVERALL_TABLE", overall_table).replace("REBUTTAL_TABLE", rebuttal_table))


def main():

  with open('disapere/rebuttal_review_salience.json', 'r') as f:
    obj = json.load(f)

    for review in obj:
      write_review_to_html(review, 'disapere/')


if __name__ == "__main__":
  main()
