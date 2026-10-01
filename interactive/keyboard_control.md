Do the following for the keyboard implementation. Do not change any of the underlying physics, how scenarios progress, etc. This is purely how controls (e.g. buttons, or other components triggers in the scene). In some cases I might ask additional elements to be added. These should not change anything about the main envs. Remember env files remain constant, all conditions under robot control remain the same and the same dynamics apply to the keyboard scenarios:


catch_marbles_trapdoors: Keys 1,2,3,or 4  OR mouse click on the corresponding key. the key is triggered by either keyboard or mouse and the corresponding door is opened

Catch_ramp_ball: mouse clicks on the empty space on the table and the cup teleports to that location. It can only occur once, after that mouse does not work

catch_cuboid: click on the cuboid (once per cuboid). if the cuboid is above the box, it comes out and stays outside; if clicked while below the box surface it is a failure


Catch_shelf_marble: move the bowl by either using left or right arrow on the keyboard or by mouse clicking on each button


Catch_valley_ball: mouse click on an area on the table and the box teleport there. IT can only happen once after that mouse does not work

Stop_Valle_ball: Mouse click on the location and bat moves there (keep the height standard at a place the can potential stop the ball

Cook_meat and cook_meat_timer: meat starts in the pan. the keys: if only a single key, pressing space, if two keys by left and right keys

put_cup_belt: mouse click on the belt and cup teleport there

Dispense_gummy: either mouse click on the keys, or use space to dispense and use right and left keys to move the bowl left or right


punch_dual_punch: left and right keys trigger clicks

save goal: click on the location and the goal keeper appears

hit_target: click via mouse on the target and then attach the dart. If click is on the blocker, then dart hit the blocker and fails

load_train: marble will be hovering over the low section of the train, and press space to release at the right time

marble_maze_shelf: keys pressed by arrow left/right keys or mouse click (as before do not keep them pressed if the key press or mouse click is not held)

pack_fruits: mouse click on the chosesn fruit (it gets highlighted by making the corresponding color lighter) and then select the basket by mouse indicating where it should go



pick_Ripe_apple: either click the corresponding arrow key to pick (whether the apple is left or right), then to apple goes over the center position of the initial position of the basket. It will be released by pressing space. Using mouse click, the corresponding apple, then apple moves over the basket and the can be clicked again to release. So apple can be picked by mouse click or arrow, and released with the second mouse click or space

Place_Block_belt: map mouse click on the part of belt to teleport the cube. map the x/y and adjust the z to the belt height accrodingly

play_billiard: click on the screen (map the x/y and the height will be adjusted to the ball). Map mouse clocik to the tip of the stick. click left/right keys to rotate the stick clockwise/coutnerclockwise. and press space to hit

Control_quality: press left or right or mouse click to trigger (same as before  do not keep triggered it the mouse is not being pressed)

drop_ball_hole: Click on the surface of the rotating platform, and then the ball drops on that location, from z_max elevate height

sort_apples_belt: using left and right keys to rotate the gate. press left and right together to open the gate for spoiled apple

whack moles: same as catch cuboid, click on moles and if they are outside, trigger a hit


#########################################
Do the following for the keyboard implementation. Do not change any of the underlying physics, how scenarios progress, etc. This is purely how controls (e.g. buttons, or other components triggers in the scene). In some cases I might ask additional elements to be added. These should not change anything about the main envs. Remember env files remain constant, all conditions under robot control remain the same and the same dynamics apply to the keyboard scenarios:


trap_bug: click on a spot on the table and the trap drops from 4 cm from the above. 

boil_milk: mouse click on the knob or space bar to turn the knob and the turn on/off the stove

fill_coffee jar: press 1,2,3, or 4 for different degree of filling

pour_beer: click and hold on the button or use space bar to push the button

cook_food and cook_food_timer: the food starts inside the pan. Knob trigger the same as boil milk

measure ingredient: click on the button or press space to turn on and off the nozzle

make_soup: click somewhere with mouse. That identifies where center of the top of the board should be. Place the board there 2 cm above the pot for z axis. use right and left arrows to tilt right and left.


catch_cup and catch_mouse object drop: click on the table and teleport pillow/box there on the surface of the table

stop_ball: click once on the table to place a U-shaped, gripper-like bridge. The ball must physically hit the bridge and settle on the table; clicking no longer stops the ball directly.

clean_table: sponge hovers 5 cm over the table. where clicked on the table by mouse, the sponge makes contact with the table. if over stain cleans as usual


task grouping


Keybaord tutorial

KEyboard Tutorial: I want a new tutorial for the keybnoard. There are 4 parts: Buttons, Placement, Base, Household. Under base gui, household is grayed out and not accessible, and under household, the base is grayed out

Buttons:
task 1-1: 3 colored keys each with a lamp above them. 
        Window setup: 
            Inst: turn on the lamps
            image of 1,2, 3 keys
            press numbers to turn the lamps".
    Once all lamps are on, progress to 1-2
    
    1-2: same setup with lamps on. 
        window setup: 
        inst: turn off the lamps
        a pic of a mouse with left click highlighted
         Click on the buttons using mouse

Task 2-1: Reset the setup. now have two keys one on the left and one on the right with arrows on them and a bowl above them (similar to dispense gummy setup). There are two red circles right above each task. the goal is the bowl to be moved to the circles. 
    Window setup:
        inst: move the bowl left and right over the cirles
        Image of right and left arrows
        Pressed arrows keys to move the bowl
    Once both circles turned green progrees to task 2-2
    
Task 2-2: same setup but reset the bowl between the circles

    Window setup:
        inst: move the bowl left and right over the cirles
        a pic of a mouse with left click highlighted
         Click on the buttons using mouse
Task 3-1: a single on/off key (with proper logo)
    Window setup:
        inst: Turn on/off the key
        Image of space
        Press space to trigger the key
    After one circle on/off move to to 3-2
    
    3-2 Same thing with mouse (use the same pattern as before)
    
Task 4-1 and 4-2: do the same with push buttons

In each task, *-1 mouse should not trigger the keys and *-2 keyboard should not


Placement: 

task 1: show a cup on one side and a green circle on the other side. 
    Window setup: 
        Inst: place the cup on the green area (no part should be outside)
        Show picture of mouse with left button highlighted
        Click on the green region to move the cup
    Progress once the cup is fully inside the green area
Task 2: put an apple on one side and a box on the other side. Pick and place should be the same mechanism as in pack_fruits. Once clicked highlighted

    Window setup: 
        Inst: Move the apple inside the box
        Show picture of mouse with left button highlighted
        Click on the apple, once highlighted, then press the box
    Task finished once the apple is inside the box. If the apple dropped of the table for any reasons and not selectable, reset the setup
    

Base: 

tasl 1: same as placement task 2 under cuttons. This time the setup of the buttons is the same as the dispense gummy, two arrow buttons on the left and trigger button on the right. On the path of the bowl, randomly select a red circle. The user should move the bowl to that regions and press trigger key to turn the circle green


    1-1: Do so with two arrow keys and space
    1-2 : only mouse
    Do the window content automatically based on previous instructions
    in 1-1 mouse does not work and in 1-2 keys won't work
    
Task 2: Show a setup similar to sort fruits: two keys and a gate on the table. The task is to open the gate by simultaneous pressing both keys, that are controlled by left and right keys  (no mouse control here)

Task 3: This is similar to the billiard task. Have a stick (just a simple stick). There is a green circle. The user should click the circle to place the tip of the stick and rotate to the left and right using left and right keys


Household:

Task 1: A moving ball and the task is to click on it to stop
task 2: A knob and a lamp, the task is to click on the knob to turn the lamp on and off
task 3: Similar to task 3 in base. Different is that instead of a stick now there is a wide cuboide (like cutting boarD). so when clicked on green zone, the board hover over it, and then the user uses left or right keys to tilt left and right

