# Pups Where They Don't Belong
### Bot that posts to sub.club

[sub.club](https://sub.club) is a creator payments platform for the fediverse, this bot uses the the sub.club API to publish posts to the service (API requests can be found in [this file](https://github.com/TomCasavant/PupsWhereTheyDontBelong/blob/main/subclub.py))

The bot uses object detection with tensor to extract a cutout of a dog from [Dog CEO](https://dog.ceo/), then pastes the dog onto an image from [Mapillary](https://www.mapillary.com/) to show these pets travelling the world.
