# Data and Upstream Provenance

Data must be obtained separately. This repository and its checkpoint bundle do
not include dermoscopy images, patient identifiers, split tables, or per-lesion
predictions. Exact file duplicates were handled by cryptographic hashes; visually
similar lesions must not be deleted as presumed duplicates.

The retained model used these dataset sources:

- [HAM10000, official Harvard Dataverse record](https://doi.org/10.7910/DVN/DBW86T),
  Philipp Tschandl and colleagues. Consult the dataset's current terms, including
  noncommercial restrictions; this repository does not relicense it.
- [BCN20000, official Figshare release](https://doi.org/10.6084/m9.figshare.24140028),
  the BCN20000 authors. Preserve the attribution and terms of the files obtained.
- [DERM12345, official Harvard Dataverse record](https://doi.org/10.7910/DVN/DAXZ7P),
  and its [dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11604664/).

Other experiments used SIIM-ISIC 2020, PH2, and/or the
[PanDerm research encoder](https://github.com/SiyuanYan1/PanDerm). Their results
do not replace the retained checkpoint. No PanDerm source or pretrained binary
is bundled. Check its upstream noncommercial/research restrictions before use;
do not infer permission to redistribute adapted weights from this repository.

The retained backbone uses the Keras EfficientNetV2 implementation. TensorFlow,
Keras, NumPy, SciPy, scikit-learn, and other dependencies retain their own licenses.
See [LICENSE_STATUS.md](LICENSE_STATUS.md) for the separate source-code and model
weight licensing status. This provenance note is not a legal determination that
all possible downstream uses or redistribution are permitted.
