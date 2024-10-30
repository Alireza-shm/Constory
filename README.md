# ConStory: Automatic story investigator of public perception on the mega urban infrastructure project 

A project focused on extracting the public opinions of North Houston Highway Improvement Project (IH-45) from Twitter

 *  Copyright (C) 2022  The University of Texas at Arlington
 *  Copyright (C) 2022  HBE: The Humanized Built Environment, (https://hubilab.uta.edu/)
 *  Copyright (C) 2022  Alireza Shamshiri, Kyeong Rok Ryu, Steven McCullough, and June Young Park

## Citation of this project

- Alireza Shamshiri, Kyeong Rok Ryu, Steven McCullough, and June Young Park. 2022. ConStory: Automatic story investigator of public perception on the mega urban infrastructure project: poster abstract. In Proceedings of the 9th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation (BuildSys '22). Association for Computing Machinery, New York, NY, USA, 293–294. https://doi.org/10.1145/3563357.3567751

## Data collection & Preprocessing

- 4103 Scrapped Tweets from Twitter 
- Text preprocessing has been done using Natural Language Toolkit (NLTK)

## Topic Modeling

- Topic Modeling Performed Using Gibbs sampling algorithm for a Dirichlet Mixture Model (https://github.com/rwalk/gsdmm)

## Temporal Analysis of Tweets

<p align="center">
		<img align="center" src="https://github.com/Alireza-shm/Constory/blob/main/Images/tA.jpg" "height="500" width="500" />
</p>
			- Number of Tweets between 2008 and 2021, Bottom: Most used words in posted tweets for each phase

## Topic Modeling Results

<p align="center">
		<img align="center" src="https://github.com/Alireza-shm/Constory/blob/main/Images/Tr.jpg" "height="500" width="500" />
</p>
- Weights of extracted topics based on the appearing in the number of tweets posted in each year

- Topic Weights = Number of topics assigned to each tweet / Each year tweets contain that topic

## Citation

If you liked our paper, please consider citing it
```bibtex
@inproceedings{shamshiri2022constory,
  title={ConStory: Automatic story investigator of public perception on the mega urban infrastructure project},
  author={Shamshiri, Alireza and Ryu, Kyeong Rok and McCullough, Steven and Park, June Young},
  booktitle={Proceedings of the 9th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation},
  pages={293--294},
  year={2022}
}
```
