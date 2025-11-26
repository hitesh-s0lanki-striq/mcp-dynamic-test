# macOS WiFi password retrieval
import subprocess
import re

# Find WiFi interface (usually en0 or en1)
def get_wifi_interface():
    try:
        result = subprocess.check_output(['networksetup', '-listallhardwareports'])
        result = result.decode('utf-8', errors='backslashreplace')
        
        lines = result.split('\n')
        for i, line in enumerate(lines):
            if 'Wi-Fi' in line or 'AirPort' in line:
                # Next line should contain the device
                if i + 1 < len(lines):
                    device_line = lines[i + 1]
                    match = re.search(r'Device: (\w+)', device_line)
                    if match:
                        return match.group(1)
    except Exception as e:
        print(f"Error finding WiFi interface: {e}")
    
    # Fallback to common interface names
    for interface in ['en0', 'en1']:
        try:
            subprocess.check_output(['networksetup', '-listpreferredwirelessnetworks', interface], 
                                  stderr=subprocess.DEVNULL)
            return interface
        except:
            continue
    
    return None

# Get list of preferred WiFi networks
def get_wifi_networks(interface):
    try:
        result = subprocess.check_output(['networksetup', '-listpreferredwirelessnetworks', interface])
        result = result.decode('utf-8', errors='backslashreplace')
        
        networks = []
        for line in result.split('\n'):
            line = line.strip()
            if line and not line.startswith('Preferred'):
                # Remove leading numbers and dots (e.g., "1. NetworkName")
                network_name = re.sub(r'^\d+\.\s*', '', line)
                if network_name:
                    networks.append(network_name)
        return networks
    except Exception as e:
        print(f"Error listing WiFi networks: {e}")
        return []

# Get password for a WiFi network
def get_wifi_password(network_name):
    try:
        result = subprocess.check_output(['security', 'find-generic-password', 
                                        '-wa', network_name],
                                       stderr=subprocess.DEVNULL)
        password = result.decode('utf-8', errors='backslashreplace').strip()
        return password
    except Exception as e:
        return None

# Main execution
if __name__ == "__main__":
    interface = get_wifi_interface()
    
    if not interface:
        print("Could not find WiFi interface")
        exit(1)
    
    print(f"Using WiFi interface: {interface}\n")
    
    networks = get_wifi_networks(interface)
    
    if not networks:
        print("No preferred WiFi networks found")
        exit(1)
    
    for network_name in networks:
        password = get_wifi_password(network_name)
        if password:
            print(f"The password of {network_name} is {password}")
        else:
            print(f"Could not retrieve password for {network_name} (may require admin privileges)")
            
# Arn@7602
# M@ze@123