"""Tracks the players with a bounding box representation using YOLO"""

import cv2
from ultralytics import YOLO
import time
import numpy as np

class PlayerTracker:
    """
    This class creates a real-time stream that tracks football players.

    Attributes:
    path_to_model: the path to the .pt model to be loaded into YOLO
    model: The loaded YOLO model
    team1: The home team
    team2: The away team
    side: 0 for home team 
    """
    
    def __init__(self, path_to_model: str, team1: dict[tuple[str,str]], team2: dict[tuple[str,str]])-> None:
        self.model = YOLO(path_to_model) 
        self.model = self.model.to("mps")
        self.team1 = team1 # Home
        self.team2 = team2 # Away
        self.side = 1

    def assign_teams(self, box, frame):
        """Assign teams to the players based on jersey color. 

           Params:
                box, frame: These are the image frame and the bounding box of the selected player

            Returns:
                Returns 0 for home team, 1 for away team, 
                2 for home keeper, and 3 for away keeper.   
        """
        if isinstance(box, list):
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        else:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
    
        shrink_factor = 5
        x1 = int(x1 + shrink_factor)
        y1 = int(y1 + shrink_factor*3)
        x2 = int(x2 - shrink_factor)
        y2 = int(y2 - shrink_factor*3)
        cropped_image = frame[y1:y2, x1:x2]

        avg_color = np.mean(cropped_image, axis=(0, 1))

        if avg_color[2] < 92: # Blue
            return 1
        else:
            return 0
        
        # TODO: Still need to impliment goalkeeper colors

    def assign_players_kickoff(self, locations: list[list[int]], results, frame)-> None:
        """Assigns all visable players to their names at kickoff. Modifies the team1 and team2 attributes."""

        if self.side == 0: # Home side on the left
            pass
        else: # Home side on the right
            team1_keys = list(self.team1.keys())
            team2_keys = list(self.team2.keys())
            team2_count = 0
            for i in range(len(locations)):
                if self.assign_teams(locations[i], frame) == 1: # team2
                    team2_count += 1
                    self.team2[team2_keys[i]] = (locations[i], 0)
                else: # team1
                    self.team1[team1_keys[i - team2_count]] = (locations[i], 0)


            
        
    
    def print_single_player(self, box, frame):
        if isinstance(box, list):
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        else:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

        cropped_image = frame[y1:y2, x1:x2]

        success = cv2.imwrite("active_player.png", cropped_image)

    def assign_players_continuous(self):
        """Assigns any new players that enter the screen to their names."""


    def assign_possesion(self):
        """Assigns 1 as the second tuple entry within the squad dictionary if player is in possesion."""


    def class_normalization(self):
        """Normalizes any jitter within the class parameter"""

    
    def get_active_coordinates_single(self, tensor)-> list[int]:
        """Converts tensor xy coordinates to simple x,y coords as integers"""
        return [round(item) for item in tensor[0].tolist() if item[-1] == 2]
    
    def get_active_coordinates(self, results):
        coords = []
        for box in results.boxes:
            if box.data[0].tolist()[-1] == 2 or box.data[0].tolist()[-1] == 1:
                coords.append([round(item) for item in box.data[0].tolist()])
        coords.sort(key=lambda box: box[0])  # Sort by x1 coordinate
        return coords

    def run_player_tracker(self, skip_frames: float = 4)-> None:
        """Runs the player tracker in real time. 

            Params:
                - skip_frame: this controlls how many frames are skipped (default is 2, meaning every other frame is skipped)
                - video_source: TBD
        """
        
        video_path = "/Users/gabriel/Documents/GitHub/EAFC-ML-Remaster/data/video_clips/arsenal_chelsea_EAFC_30sec.mp4"
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("ERROR: Video stream not opened properly.")
            exit()

        frame_counter = 0
        iterations = 0
        while True:
            frame_counter += 1
            succ, frame = cap.read()

            if not succ:
                print("Process Complete.")
                break

            if frame_counter == skip_frames:
                frame_counter = 0
                results = self.model.predict(frame) # Run single frame inference w/ YOLO
                results = results[0]
                
                coords = self.get_active_coordinates(results)
                if iterations == 0:
                    self.assign_players_kickoff(coords, results, frame) # initalize teams
                    iterations = 1
                
                # self.print_single_player([983, 122, 1006, 187, 1, 2], frame)
                # exit(0)
                
                for box in results.boxes:

                    # active_player = self.get_active_coordinates_single(box.data)

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())

                    # Draw label
                    # Find the player's name:
                    name = "NOT FOUND"
                    for i in range(5): 
                        temp = sum(self.team1[(list(self.team1.keys())[i])][0][0:4])
                        tolerance = abs((x1 + y1 + x2 + y2) - temp)
                        if tolerance < 20:
                            name = list(self.team1.keys())[i]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                time.sleep(1)
                cv2.imshow("EAFC25 ML Remaster", frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # End process
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":

    # Arsenal Squad
    arsenal = {
    "Mikel Merino": (),
    "Ethan Nwaneri": (),
    "Leandro Trossard": (),
    "Martin Odegaard": (),
    "Jurrian Timber": (),
    "Thomas Partey": (),
    "Declan Rice": (),
    "William Saliba": (),
    "Gabriel": (),
    "Miles Lewis-Skelly": (),
    "David Raya": ()
    }

    # Chelsea Squad
    chelsea = {
    "Marc Cucurella": (),
    "Pedro Neto": (),
    "Moises Caicedo": (),
    "Cole Palmer": (),
    "Nicholas Jackson": (),
    "Christopher Nkunku": (),
    "Enzo Fernandez": (),
    "Reece James": (),
    "Trevor Chalobah": (),
    "Levi Colwill": (),
    "Alexis Sanchez": ()
    }


    pt = PlayerTracker("models/yolo_v5_pretrained.pt", team1=arsenal, team2=chelsea)
    pt.run_player_tracker()
