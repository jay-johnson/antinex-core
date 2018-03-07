class MockMessage:

    def __init__(
            self,
            exchange=None,
            routing_key=None,
            queue=None):
        """__init__

        :param exchange: mock exchange name
        :param routing_key: mock routing key
        :param queue: mock queue
        """
        self.state = "NOTRUN"
        self.delivery_info = {
            "exchange": exchange,
            "routing_key": routing_key,
            "queue": queue
        }
    # end of __init__

    def get_exchange(
            self):
        """get_exchange"""
        return self.delivery_info["exchange"]
    # end of get_exchange

    def get_routing_key(
            self):
        """get_routing_key"""
        return self.delivery_info["routing_key"]
    # end of get_routing_key

    def get_queue(
            self):
        """get_queue"""
        return self.delivery_info["queue"]
    # end of get_queue

    def ack(self):
        """ack"""
        self.state = "ACK"
    # end of ack

    def reject(self):
        """reject"""
        self.state = "REJECT"
    # end of reject

    def requeue(self):
        """requeue"""
        self.state = "REQUEUE"
    # end of requeue

# end of MockMessage
