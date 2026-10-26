---
doc_id: Packet_switching__p006
title: Packet switching
collection: squad_v11_b5
source: SQuAD v1.1 dev
license: CC BY-SA 4.0 (SQuAD, Rajpurkar et al. 2016)
---

Connection-oriented transmission requires a setup phase in each involved node before any packet is transferred to establish the parameters of communication. The packets include a connection identifier rather than address information and are negotiated between endpoints so that they are delivered in order and with error checking. Address information is only transferred to each node during the connection set-up phase, when the route to the destination is discovered and an entry is added to the switching table in each network node through which the connection passes. The signaling protocols used allow the application to specify its requirements and discover link parameters. Acceptable values for service parameters may be negotiated. Routing a packet requires the node to look up the connection id in a table. The packet header can be small, as it only needs to contain this code and any information, such as length, timestamp, or sequence number, which is different for different packets.
