import numpy as np
import tensprflow as tf




# see what's in video
# decide what to do with clip
# Add all clips together
#### Get probability of what clips should come before and after then
#### Add Clips together

#clip attribute inputs: {length: int,
                       # type: int,
                       # intensity: int,
                       # what's happening: string turned to integers and then summed (letters represent numbers for nn (e.g. "hello" = 8, 5, 12, 12, 15) then summed (8+5+12+12+15 = 52))
                       # audio: [monolog(1), more than one person talking/cheering(2), music in the background(3), misc(4)]
                       # time audio mentions something happening in clip: float
                       # }

#clip outputs {
    # type of clip before: int
    # type of clip after: int
    # clip_section: [begining(1), filler(2), end(3)]
    # clip_effect_start_time: float
    # clip_effect_end_time: float
    # main_monolog: boolean,
    # play_audio: boolean,
    # top_three_times_to_play_clip: [1st: float, 2nd:float, 3rd:float]
    # clip start: float
    # clip end: float
    # audio start: float(0.0 if no playable audio, -float if audio starts beforeclip starts)
    # audio end: float(0.0 if no playable audio, )
# }


#training
# determine audio with deepnet
# plug in clips from every 2 seconds and get image of what's happening that detetermines clip name



#adding together{
    #start at edges and work inward after all clips have been edited individually.
#}
