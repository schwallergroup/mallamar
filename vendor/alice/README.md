# Vendored ALICE runtime

Mallamar installs the Python package in `src/alice` through its root environment and packaging. This copy contains the runtime modules needed for Bayesian optimization; use [the upstream ALICE repository](https://github.com/schwallergroup/alice) for its standalone training runner, benchmark configurations and datasets.

Mallamar-specific changes are recorded in the root `NOTICE`. The upstream license is retained in `LICENSE`.

## Citation

```bibtex
@misc{sin2026dynamic,
  title         = {Dynamic language model representations for multi-objective reaction optimisation},
  author        = {Sin, Joshua W. and Segura, David Ming and Ranković, Bojana and Chau, Siu Lun and Lutz, Marius D. R. and Anelli, Andrea and Burwood, Ryan P. and Püntener, Kurt and Notheis, Maximilian J. and Bigler, Raphael and Schwaller, Philippe},
  year          = {2026},
  eprint        = {2609.11790},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2609.11790},
  url           = {https://arxiv.org/abs/2609.11790}
}
```

---

## License

This project is licensed under the **Apache 2.0 License**. See the `LICENSE` file for details.

---

## Acknowledgements

This work was supported by NCCR Catalysis (grant no. 225147), a National Centre
of Competence in Research funded by the Swiss National Science Foundation, and
by the Swiss National Science Foundation (SNSF) [226509].