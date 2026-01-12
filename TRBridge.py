from time import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import torch
from model import VisionEncoder, RecurrentActorNetwork

class TRAgent(Node):
    def __init__(self):
        super().__init__('tra_agent')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.drive = self.create_publisher(AckermannDriveStamped, 'drive', 10)
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10
        )
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',
            self.odom_callback,
            10
        )
        
        self.velocities = None
        
        encoder = VisionEncoder()
        model = RecurrentActorNetwork(
                3, 
                2, 
                encoder=encoder,
                lstm_hidden_size=256,
                memory_length=20,
                memory_stride=5
            )
        
        self.model = torch.jit.load('actor_BrandsHatch_jit.pt').to(self.device)
        
        self.actor_lstm_buffer = [self.model.create_observation_buffer(1, self.device)]
        self.actor_hidden = [self.model.get_init_hidden(1, self.device, transpose=True)]
        
        self.params = {
               's_min': -0.4189,
               's_max': 0.4189,
               'v_min':-5.0,
               'v_max': 20.0,
               }
        
        
    def scan_callback(self, msg):
        scan_data = msg.ranges
        
        if self.velocities is None:
            return
        
        scan_tensor = torch.tensor(scan_data, dtype=torch.float32).to(self.device)
        scan_tensor = scan_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        state_tensor = torch.tensor(self.velocities, dtype=torch.float32).to(self.device)
        state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)  # Add batch dimension
        
        scaled_action = self.get_action_and_value(scan_tensor, state_tensor)
        
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = scaled_action[0, 0].item()
        drive_msg.drive.speed = scaled_action[0, 1].item()
        
        self.drive.publish(drive_msg)
        
    def odom_callback(self, msg):
        self.velocities = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z]
        
    def get_action_and_value(self, scan_tensor, state_tensor):
        with torch.no_grad():
            # Get action from actor
            loc, _, actor_new_buffer, actor_new_hidden_h, actor_new_hidden_c = self.model(
                scan_tensor[0],
                state_tensor[0],
                self.actor_lstm_buffer[-1],
                self.actor_hidden[-1][0],
                self.actor_hidden[-1][1]
            )

            # Create distribution and sample action
            action = loc  # Use mean for deterministic policy
    
            self.actor_lstm_buffer.append(actor_new_buffer)
            self.actor_hidden.append((actor_new_hidden_h, actor_new_hidden_c))
            
            # Scale to environment's action space
            steer_scale = (self.params['s_max'] - self.params['s_min']) / 2
            steer_shift = (self.params['s_max'] + self.params['s_min']) / 2
            speed_scale = (self.params['v_max'] - self.params['v_min']) / 2
            speed_shift = (self.params['v_max'] + self.params['v_min']) / 2
            
            steering = steer_scale * action[..., 0].unsqueeze(-1) + steer_shift
            speed = speed_scale * action[..., 1].unsqueeze(-1) + speed_shift
            
            scaled_action = torch.cat((steering, speed), dim=-1)
            
        return scaled_action
    
    def transfer_weights(self, path, network):
        checkpoint = torch.load(path)
        
        prefix = "0.module."

        state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith(prefix):
                new_key = k[len(prefix):]
                state_dict[new_key] = v
            else: state_dict[k] = v

        if state_dict:
            network.load_state_dict(state_dict)
            print("Successfully loaded pre-trained weights!")
            
        return network

def main(args=None):
    try:
        rclpy.init(args=args)
        minimal_publisher = TRAgent()
        rclpy.spin(minimal_publisher)
        minimal_publisher.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = 0.0
        drive_msg.drive.speed = 0.0
        minimal_publisher.drive.publish(drive_msg)
        rclpy.shutdown()

if __name__ == '__main__':
    main()