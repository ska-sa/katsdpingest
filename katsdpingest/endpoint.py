import socket
import struct
import netifaces

class Endpoint(object):
    """A TCP or UDP endpoint consisting of a host and a port"""

    def __init__(self, host, port):
        self.host = host
        self.port = port

    def __eq__(self, other):
        return self.host == other.host and self.port == other.port

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        if ':' in self.host:
            # IPv6 address - escape it
            return '[{0}]:{1}'.format(self.host, self.port)
        else:
            return '{0}:{1}'.format(self.host, self.port)

    def __repr__(self):
        return 'Endpoint({0!r}, {1!r})'.format(self.host, self.port)

    def __iter__(self):
        """Support `tuple(endpoint)` for passing to a socket function"""
        return iter((self.host, self.port))

    def multicast_subscribe(self, sock):
        """If the address is an IPv4 multicast address, subscribe to the group
        on `sock`. Return `True` if the host is a multicast address.
        """
        try:
            raw = socket.inet_aton(self.host)
        except socket.error:
            return False
        else:
            # IPv4 multicast is the range 224.0.0.0 - 239.255.255.255
            if raw[0] >= chr(224) and raw[0] <= chr(239):
                for iface in netifaces.interfaces():
                    for addr in netifaces.ifaddresses(iface).get(netifaces.AF_INET, []):
                        # Skip point-to-point links (includes loopback)
                        if 'peer' in addr:
                            continue
                        if_raw = socket.inet_aton(addr['addr'])
                        mreq = struct.pack("4s4s", raw, if_raw)
                        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
                        break  # Should only need to subscribe once per interface
                return True
            else:
                return False


