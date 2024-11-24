# model-api
 model api with flask

<!-- section generate summary pada mbart -->
        # Generate summary
        with torch.no_grad():
            # Opsi 1: Menggunakan beam search dengan diversity
            # Opsi 1 akan menghasilkan output yang lebih deterministik tapi beragam antar beam groups
            summary_ids = self.model.generate(
                inputs["input_ids"],
                num_beams=4,
                min_length=min_length,
                max_length=max_length,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.5,
                num_beam_groups=4,
                diversity_penalty=0.5,
                do_sample=False  # Harus False ketika menggunakan diversity_penalty
            )

            # # Opsi 2: Menggunakan sampling 
            # Opsi 2 akan menghasilkan output yang lebih kreatif dan bervariasi setiap kali generate
            # summary_ids = self.model.generate(
            #     inputs["input_ids"],
            #     num_beams=4,
            #     min_length=min_length,
            #     max_length=max_length,
            #     early_stopping=True,
            #     no_repeat_ngram_size=3,
            #     length_penalty=1.5,
            #     temperature=temperature,
            #     do_sample=True,
            #     top_k=50,
            #     top_p=0.9,
            #     repetition_penalty=repetition_penalty
            # )

<!-- Parameter yang bisa diatur dalam function generate summary mbart -->
# Untuk hasil yang lebih panjang
length_penalty=2.0  # > 1.0 mendorong kalimat lebih panjang

# Untuk hasil yang lebih pendek
length_penalty=0.8  # < 1.0 mendorong kalimat lebih pendek

# Untuk hasil yang lebih beragam
num_beam_groups=4
diversity_penalty=0.5

# Untuk hasil yang lebih fokus
num_beams=6
no_repeat_ngram_size=2
