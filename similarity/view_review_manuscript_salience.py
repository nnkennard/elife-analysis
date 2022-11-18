import json
import pandas as pd
import numpy as np

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
               FIRST_TABLE
            </div>
            <div class="column is-half">
               SECOND_TABLE
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


def write_review_and_manuscript_to_html(review, directory):
    build_tables(
        review["manuscript_sentences"],
        review["review_sentences"],
        review["result"],
        review["review_id"],
        directory,
    )


def build_tables(first_sentences, second_sentences, result, review_id, directory):
    first_table = (
        pd.DataFrame.from_dict([{"first_sentence": sent} for sent in first_sentences])
        .to_html()
        .replace(
            '<table border="1" class="dataframe">', '<table border="1" class="table">'
        )
        .replace("\n", "")
    )
    overall_table_dicts = []
    second_table_dicts = []

    for first_i, first_sent in enumerate(first_sentences):
        second_sent_values = {"sbert": [], "jaccard": []}
        for second_i, second_sent in enumerate(second_sentences):
            sbert_val = result["sbert"][second_i][first_i]
            jaccard_val = result["jaccard"][second_i][first_i]
            second_sent_values["sbert"].append(sbert_val)
            second_sent_values["jaccard"].append(jaccard_val)
            overall_table_dicts.append(
                {
                    "first_sentence": first_sent,
                    "second_sentence": second_sent,
                    "sbert": sbert_val,
                    "jaccard": jaccard_val,
                }
            )
        second_table_dicts.append(
            {
                "second_sentence": second_sent,
                "sbert_mean": sum(second_sent_values["sbert"])
                / len(second_sent_values["sbert"]),
                "jaccard_mean": sum(second_sent_values["jaccard"])
                / len(second_sent_values["jaccard"]),
            }
        )
    overall_table = (
        pd.DataFrame.from_dict(overall_table_dicts)
        .to_html()
        .replace(
            '<table border="1" class="dataframe">',
            '<table border="1" class="fancytable">',
        )
        .replace("\n", "")
    )
    second_table = (
        pd.DataFrame.from_dict(second_table_dicts)
        .to_html()
        .replace(
            '<table border="1" class="dataframe">',
            '<table border="1" class="fancytable">',
        )
        .replace("\n", "")
    )
    with open(f"{directory}/{review_id}.html", "w") as f:
        f.write(
            HTML_TEMPLATE.replace("\n", "")
            .replace("FIRST_TABLE", first_table)
            .replace("OVERALL_TABLE", overall_table)
            .replace("SECOND_TABLE", second_table)
        )


def main():

    with open("disapere_results/review_manuscript_salience.json", "r") as f:
        obj = json.load(f)

        for review in obj:
            write_review_and_manuscript_to_html(
                review, "disapere_results/review_manuscript_results/"
            )


if __name__ == "__main__":
    main()
