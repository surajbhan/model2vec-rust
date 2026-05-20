import json
import struct

def inspect_safetensors(file_path):
    with open(file_path, 'rb') as f:
        # Read header size
        header_size_bytes = f.read(8)
        if len(header_size_bytes) < 8:
            print("Invalid safetensors file: file too short")
            return
        header_size = struct.unpack('<Q', header_size_bytes)[0]
        
        # Read header JSON
        header_json_bytes = f.read(header_size)
        if len(header_json_bytes) < header_size:
            print("Invalid safetensors file: header size mismatch")
            return
        
        header = json.loads(header_json_bytes.decode('utf-8'))
        print("Safetensors Keys and Metadata:")
        for k, v in header.items():
            if k == "__metadata__":
                print(f"Metadata: {v}")
            else:
                print(f"Tensor: {k} -> {v}")

if __name__ == '__main__':
    inspect_safetensors('model/model.safetensors')
