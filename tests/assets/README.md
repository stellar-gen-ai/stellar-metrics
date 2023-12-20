# Testing Assets

The assets are used to verify the expected behavior of the metrics. They are used to simulate how the metrics are expected to be used. Due to [license restrictions from Celeb](https://github.com/switchablenorms/CelebAMask-HQ?tab=readme-ov-file#license-and-citation) we use freely available stock photos.

The photos are taken from:

* [000-0](https://www.pexels.com/photo/woman-at-beach-3324738/)
* [000-1](https://iordanis.me)
* [199-0](https://www.pexels.com/photo/woman-taking-photo-of-herself-near-parked-vehicles-1085517/)
* [199-1](https://www.pexels.com/photo/smiling-man-wearing-gray-hat-3030332/)

The goal of the tests and assets of this folder is that:

1. We present a folder structure for the expected format of the dataset [mock_stellar_dataset](mock_stellar_dataset)
2. Present a folder structure for the expected format of the generated images [stellar_net](stellar_net)


## Output Example

| metric                |     value |         std |
|:----------------------|----------:|------------:|
| attr                  | 0.683333  | nan         |
| object_faithfulness   | 0.221197  |   0.162911  |
| clip_t                | 0.42505   |   0.0488983 |
| clip_n                | 0.195358  |   0.0337628 |
| clip_i                | 0.398792  |   0.0958932 |
| identity_preservation | 0.325054  |   0.132854  |
| identity_stability    | 0.0265809 |   0.151951  |